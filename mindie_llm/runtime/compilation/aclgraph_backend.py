# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Optional, Any, Dict
import torch
from mindie_llm.runtime.model_runner.forward_context import get_forward_context
from mindie_llm.utils.log.logging import logger

# Status indicating whether aclgraph can be captured.
aclgraph_capturing_enable = False
# Reuse this memory pool across all acl graph runners.
_global_graph_memory_pool = None


def get_global_graph_memory_pool() -> Optional[Any]:
    """Get the global graph memory pool.
    
    Returns:
        The global graph memory pool, or None if not set.
    """
    return _global_graph_memory_pool


def set_global_graph_memory_pool(val: Any) -> None:
    """Set the global graph memory pool.
    
    Args:
        val: Memory pool value to set.
    """
    global _global_graph_memory_pool
    _global_graph_memory_pool = val


def validate_aclgraph_capturing_enabled() -> None:
    """Validate that ACL graph capturing is enabled.
    
    Raises:
        RuntimeError: If ACL graph capturing is not enabled.
    """
    global aclgraph_capturing_enable
    if not aclgraph_capturing_enable:
        raise RuntimeError(
            "ACL graph capturing detected at an inappropriate "
            "time. This operation is currently disabled."
        )


def set_aclgraph_capturing_enabled(enabled: bool) -> None:
    """Set the ACL graph capturing enabled status.
    
    Args:
        enabled: Whether to enable ACL graph capturing.
    """
    global aclgraph_capturing_enable
    aclgraph_capturing_enable = enabled


class AclGraphBackend:
    """Backend for ACL graph compilation and execution.
    
    This class wraps a model to enable graph mode execution using ACL graphs.
    It manages graph capture, replay, and handles both eager and graph modes.
    """
    
    def __init__(self, model: Any, max_num_tokens_per_batch: int) -> None:
        """Initialize the ACL graph backend.
        
        Args:
            model: The model to wrap with graph backend.
            max_num_tokens_per_batch: Maximum number of tokens per batch.
        """
        self.model = model
        self.graphs: Dict[Any, Optional[torch.npu.NPUGraph]] = {}
        # NOTE: only used for outputs' memory calculation
        self.output_buffer: Dict[Any, torch.Tensor] = {}
        self.input_addresses: Dict[Any, Any] = {}
        self.capture_sizes = self._get_capture_sizes(max_num_tokens_per_batch)
        self._run_eager_mode_with_padding = False

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Execute forward pass using graph mode or eager mode.
        
        In prefill stage or when num tokens is larger than max capture size,
        eager mode will be used. Otherwise, graph mode will be used.
        
        Args:
            *args: Positional arguments for model forward pass.
            **kwargs: Keyword arguments for model forward pass.
            
        Returns:
            Hidden states tensor.
        """
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        num_actual_tokens = forward_context.num_actual_tokens
        attn_metadata_dict = forward_context.attn_metadata_dict

        #In prefill stage or num tokens lager than max capture size will use eager mode
        if self._run_eager_mode_with_padding or forward_context.is_prefill\
            or num_actual_tokens > self.capture_sizes[-1]:

            hidden_state = self.model(*args, **kwargs)
            return hidden_state

        if batch_descriptor not in self.graphs:
            self.graphs[batch_descriptor] = None
        graph = self.graphs[batch_descriptor]

        if graph is None:
            # start capturing
            validate_aclgraph_capturing_enabled()
            aclgraph = torch.npu.NPUGraph()

            if get_global_graph_memory_pool() is None:
                set_global_graph_memory_pool(torch.npu.graph_pool_handle())

            forward_context.capturing = True
            with torch.npu.graph(
                npu_graph=aclgraph,
                pool=get_global_graph_memory_pool(),
                capture_error_mode="thread_local",
                auto_dispatch_capture=True
            ):
                hidden_state = self.model(*args, **kwargs)

            self.output_buffer[batch_descriptor] = hidden_state
            self.graphs[batch_descriptor] = aclgraph
            graph = aclgraph

        graph.replay()
        try:
            actual_seq_lengths_kv = attn_metadata_dict[next(iter(attn_metadata_dict))].seq_lens_list
            graph.update(
                cpu_update_input=[{"actual_seq_lengths_kv": actual_seq_lengths_kv}]
            )  # NOTE: hardcode first attn_metadata.
        except Exception as e:
            logger.error(f"Ignore graph update failure: {e}.")

        hidden_state = self.output_buffer[batch_descriptor]
        return hidden_state[:num_actual_tokens]

    @staticmethod
    def _get_capture_sizes(max_num_tokens_per_batch: int, step: int = 8) -> list[int]:
        """Get list of capture sizes for graph compilation.
        
        Generates power-of-2 sizes for small batches, then linear steps for larger batches.
        Always includes max_num_tokens_per_batch to ensure full coverage.
        
        Args:
            max_num_tokens_per_batch: Maximum number of tokens per batch.
            step: Step size for large batches (default 8, aligned to NPU memory).
            
        Returns:
            Sorted list of capture sizes, always ending with max_num_tokens_per_batch.
            
        Example:
            >>> _get_capture_sizes(20)
            [1, 2, 4, 8, 16, 20]
            >>> _get_capture_sizes(9)
            [1, 2, 4, 8, 9]
            >>> _get_capture_sizes(5)
            [1, 2, 4, 5]
        """
        if max_num_tokens_per_batch < 1:
            raise ValueError(
                f"max_num_tokens_per_batch must be >= 1, got {max_num_tokens_per_batch}"
            )
        
        # Small sizes: powers of 2
        small_sizes = [1, 2, 4]
        
        # Large sizes: linear steps (excludes max)
        large_sizes = range(step, max_num_tokens_per_batch, step)
        
        # Merge, deduplicate, ensure max included, and sort
        all_sizes = sorted({*small_sizes, *large_sizes, max_num_tokens_per_batch})
        return [s for s in all_sizes if s <= max_num_tokens_per_batch]

    def get_padded_graph_size(self, x: int) -> int:
        """Get the padded graph size for a given number of tokens.
        
        Args:
            x: Number of tokens.
            
        Returns:
            The smallest capture size that is >= x, or x if no such size exists.
        """
        return next((bs for bs in self.capture_sizes if bs >= x), x)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits from hidden states.
        
        Args:
            hidden_states: Hidden states tensor.
            
        Returns:
            Logits tensor.
        """
        return self.model.compute_logits(hidden_states)

    def maybe_gather_and_unpad_for_flashcomm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Gather and unpad hidden states for flash communication if enabled.
        
        Args:
            hidden_states: Hidden states tensor.
            
        Returns:
            Processed hidden states tensor.
        """
        return self.model.maybe_gather_and_unpad_for_flashcomm(hidden_states)

    def maybe_pad_and_gather_cross_dp_and_unpad(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pad, gather, and unpad hidden states across data parallel groups if enabled.
        
        Args:
            hidden_states: Hidden states tensor.
            
        Returns:
            Processed hidden states tensor.
        """
        return self.model.maybe_pad_and_gather_cross_dp_and_unpad(hidden_states)

    def set_eager_mode_with_padding(self, is_eager_mode_with_padding: bool):
        """A method to control whether run in eager mode or graph mode
        
        Args:
            is_eager_mode_with_padding: bool, whether to enable eager mode with padding
        """
        self._run_eager_mode_with_padding = is_eager_mode_with_padding