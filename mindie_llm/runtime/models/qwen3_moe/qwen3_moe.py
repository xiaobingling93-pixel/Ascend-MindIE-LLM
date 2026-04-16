# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Any

import torch
from torch import nn

from mindie_llm.runtime.layers.embedding.embedding import VocabParallelEmbedding, ParallelLMHead
from mindie_llm.runtime.layers.linear.linear import ReplicatedLinear
from mindie_llm.runtime.layers.normalization import RMSNorm
from mindie_llm.runtime.models.base.model import BaseModelForCausalLM
from mindie_llm.runtime.models.qwen3.qwen3 import Qwen3Attention
from mindie_llm.runtime.layers.fused_moe.experts_selector import select_experts
from mindie_llm.runtime.layers.fused_moe.fused_moe import FusedMoE
from mindie_llm.runtime.model_runner.forward_context import get_forward_context
from mindie_llm.runtime.utils.distributed import get_parallel_info_manager
from mindie_llm.runtime.layers.quantization.quantization_config_base import QuantizationConfigBase


class Qwen3MoeSparseMoeBlock(nn.Module):
    """
    Qwen3 Sparse MoE (Mixture of Experts) feed-forward module.

    This class implements the sparse mixture of experts component for Qwen3 MoE model, including:
    - Expert router logits projection via replicated linear layer
    - Top-k expert selection with softmax scoring and renormalization
    - Fused parallel computation for multiple selected experts
    - Weighted aggregation of expert outputs

    Key features:
    - Replicated gate projection layer for expert routing
    - FusedMoE layer for efficient parallel expert computation

    Attributes:
        config: Model configuration object
        prefix: Prefix for parameter naming
        quant_config: Quantization configuration (if applicable)
        topk_num: Number of experts selected per token, from config.num_experts_per_tok
        expert_num: Total number of experts, from config.num_experts
        gate: ReplicatedLinear layer for computing expert router logits
        experts: FusedMoE module encapsulating all expert feed-forward layers
    """

    def __init__(
        self,
        config,
        prefix,
        quant_config: QuantizationConfigBase = None,
    ) -> None:
        """
        Initialize the Qwen3 sparse MoE block module.

        Args:
            config: Model configuration object
            prefix: Parameter naming prefix
            quant_config: Quantization configuration (optional)
        """
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.quant_config = quant_config
        self.topk_num = config.num_experts_per_tok
        self.expert_num = config.num_experts
        self.gate = ReplicatedLinear(
            config.hidden_size,
            self.expert_num,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )
        self.experts = FusedMoE(
            num_experts=self.expert_num,
            topk_num=self.topk_num,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            suffix=["gate_proj", "down_proj", "up_proj"],
        )

    def forward(self, hidden_states):
        """
        Forward pass of the sparse MoE block module.

        Args:
            hidden_states: Input hidden states tensor

        Returns:
            torch.Tensor: Output hidden states after MoE expert computation and aggregation
        """
        router_logits = self.gate(hidden_states)
        # topk
        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=self.topk_num,
            use_grouped_topk=False,
            renormalize=True,
            topk_group=1,
            num_expert_group=1,
            scoring_func="softmax",
            routed_scaling_factor=1.0,
            e_score_correction_bias=None,
            global_num_experts=self.expert_num,
        )
        final_hidden_states = self.experts(hidden_states, topk_weights, topk_ids)
        return final_hidden_states


class Qwen3MoeLayer(nn.Module):
    """
    Qwen3 MoE Transformer layer module.

    This class implements a single transformer layer for Qwen3 MoE model, including:
    - Pre-attention RMS layer normalization
    - Self-attention mechanism with positional encoding
    - Post-attention RMS layer normalization
    - Sparse MoE feed-forward network instead of standard MLP
    - Residual connection with dual path normalization strategy

    Key features:
    - Two-stage residual connection with dedicated layer norms
    - Attention and MoE components are fully quantizable

    Attributes:
        config: Model configuration object
        prefix: Layer-specific parameter naming prefix
        layer_idx: Index of current transformer layer
        quant_config: Quantization configuration (if applicable)
        self_attn_prefix: Parameter prefix for self attention module
        self_attn: Qwen3Attention self attention module
        mlp: Qwen3MoeSparseMoeBlock sparse MoE feed-forward module
        input_layernorm: RMSNorm for attention input normalization
        post_attention_layernorm: RMSNorm for MoE input normalization
    """

    def __init__(
        self,
        config,
        prefix: str,
        layer_idx: int,
        quant_config: QuantizationConfigBase = None,
    ) -> None:
        """
        Initialize the Qwen3 MoE transformer layer module.

        Args:
            config: Model configuration object
            prefix: Root parameter naming prefix
            layer_idx: Index of current transformer layer
            quant_config: Quantization configuration (optional)
        """
        super().__init__()

        self.config = config
        self.prefix = f"{prefix}.layers.{layer_idx}"
        self.layer_idx = layer_idx
        self.quant_config = quant_config

        self.self_attn_prefix = f"{self.prefix}.self_attn"
        self.self_attn = Qwen3Attention(config, self.self_attn_prefix, quant_config=quant_config)
        self.mlp = Qwen3MoeSparseMoeBlock(config, f"{self.prefix}.mlp", quant_config=quant_config)

        self.input_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, quant_config=quant_config, prefix=f"{self.prefix}.input_layernorm"
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            quant_config=quant_config,
            prefix=f"{self.prefix}.post_attention_layernorm",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Qwen3 MoE transformer layer module.

        Args:
            positions: Positional encoding tensor for attention computation
            hidden_states: Input hidden states tensor
            residual: Residual connection tensor (None for first layer input)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of updated hidden states and residual tensor
        """
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3MoeModel(nn.Module):
    """
    Qwen3 MoE base transformer model.

    This class implements the full stack of Qwen3 MoE transformer model, including:
    - Vocabulary parallel embedding layer for input tokenization
    - Sequential stack of Qwen3MoeLayer transformer layers
    - Final RMS layer normalization for output hidden states
    - End-to-end forward pass with residual connection propagation

    Key features:
    - Vocab parallel embedding for large vocabulary sharding
    - Sequential layer execution with residual tensor passing
    - Unified final normalization for stable output representation
    - Full quantization support across all model components

    Attributes:
        config: Model configuration object
        prefix: Root parameter naming prefix
        quant_config: Quantization configuration (if applicable)
        embed_tokens: VocabParallelEmbedding input token embedding layer
        layers: ModuleList of Qwen3MoeLayer transformer layers
        norm: RMSNorm final output normalization layer
    """

    def __init__(
        self,
        config: Any,
        prefix: str = "model",
        quant_config: QuantizationConfigBase = None,
    ) -> None:
        """
        Initialize the Qwen3 MoE base transformer model.

        Args:
            config: Model configuration object
            prefix: Root parameter naming prefix (default: "model")
            quant_config: Quantization configuration (optional)
        """
        super().__init__()

        self.config = config
        self.prefix = prefix
        self.quant_config = quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        self.layers = nn.ModuleList(
            [
                Qwen3MoeLayer(config, self.prefix, layer_idx, quant_config=self.quant_config)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, quant_config=quant_config, prefix=f"{prefix}.norm")

    def forward(self, input_ids, positions):
        """
        Forward pass of the Qwen3 MoE base transformer model.

        Args:
            input_ids: Input token id tensor
            positions: Positional encoding tensor

        Returns:
            torch.Tensor: Final normalized hidden states after all transformer layers
        """
        residual = None
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class Qwen3MoeForCausalLM(BaseModelForCausalLM):
    """
    Qwen3-MoE model for causal language modeling.

    This class extends the base Qwen3-MoE model with a language modeling head for:
    - Text generation
    - Next token prediction
    - Causal language modeling tasks

    Key features:
    - LM head for token prediction
    - Integration with model runner for inference

    Attributes:
        hf_config: Hugging Face configuration object
        quant_config: Quantization configuration
        parallel_info_manager: Manager for parallelism information
        model_status: Model status information
        model: Base Qwen3 model
        lm_head: Language modeling head
    """

    def __init__(self, mindie_llm_config):
        """
        Initialize the Qwen3-MoE causal language model.

        Args:
            mindie_llm_config: MindIE LLM configuration object containing:
                - hf_config: Hugging Face configuration
                - quant_config: Quantization configuration
        """
        super().__init__(mindie_llm_config)

        self.hf_config = mindie_llm_config.hf_config
        self.quant_config = mindie_llm_config.quant_config
        self.parallel_info = get_parallel_info_manager()
        self.model = Qwen3MoeModel(
            config=mindie_llm_config.hf_config,
            prefix="model",
            quant_config=self.quant_config,
        )

        self.lm_head = ParallelLMHead(
            self.hf_config.vocab_size,
            self.hf_config.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix="lm_head",
        )

    def forward(self, input_ids, positions):
        """
        Forward pass of the model (without LM head).

        Args:
            input_ids: Input token IDs
            positions: Position indices

        Returns:
            torch.Tensor: Hidden states before LM head
        """
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        """
        Forward pass through the LM head.

        Args:
            hidden_states: Hidden states from the base model

        Returns:
            torch.Tensor: Logits for token prediction
        """
        forward_context = get_forward_context()
        lm_head_indices = forward_context.lm_head_indices
        return self.lm_head.forward(hidden_states, lm_head_indices)
