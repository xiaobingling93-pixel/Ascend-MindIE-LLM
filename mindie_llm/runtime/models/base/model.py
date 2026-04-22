# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from abc import abstractmethod

import torch
from torch import nn
import torch.nn.functional as F

from mindie_llm.runtime.models.base.model_descriptor import ModelDescriptor
from mindie_llm.runtime.config.mindie_llm_config import MindIELLMConfig
from mindie_llm.runtime.model_runner.forward_context_exp import get_forward_context
from mindie_llm.runtime.utils.distributed import get_parallel_info_manager
from mindie_llm.runtime.utils.distributed.parallel_info_manager import ParallelType


class BaseModelForCausalLM(nn.Module):
    """
    Base causalLM class, which defines the interface that every model must implement
    and the parameters that must be included.

    Attributes:
        model_descriptor (ModelDescriptor): Indicates which features are enabled for this model.
    """

    def __init__(self, mindie_llm_config: MindIELLMConfig):
        super().__init__()
        self.mindie_llm_config = mindie_llm_config
        self.model_descriptor = self._get_model_descriptor_cls().from_config(mindie_llm_config)

    @staticmethod
    def maybe_gather_and_unpad_for_flashcomm(hidden_states):
        forward_context = get_forward_context()
        if not forward_context.batch_descriptor.is_flash_comm_enabled:
            return hidden_states

        from mindie_llm.runtime.layers.linear.linear_op import maybe_all_gather_and_maybe_unpad

        hidden_states = maybe_all_gather_and_maybe_unpad(
            hidden_states, get_parallel_info_manager().get(ParallelType.ATTN_TP)
        )
        return hidden_states

    @staticmethod
    def maybe_all_gather_cp(hidden_states):
        cp = get_parallel_info_manager().get(ParallelType.ATTN_CP)
        if not cp.is_enabled():
            return hidden_states

        group_size = cp.group_size
        hidden_states_out = torch.zeros_like(hidden_states).repeat(group_size, *(1,) * (hidden_states.dim() - 1))
        torch.distributed.all_gather_into_tensor(hidden_states_out, hidden_states, group=cp.process_group)
        return hidden_states_out

    @staticmethod
    def _get_model_descriptor_cls():
        """The default method to get model_descriptor class."""
        return ModelDescriptor

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        """Abstract method to compute hidden_states."""
        pass

    @abstractmethod
    def compute_logits(self, hidden_states: torch.Tensor):
        """Abstract method to compute logits."""
        pass

    def maybe_pad_and_gather_cross_dp_and_unpad(self, hidden_states):
        # NOTE: Support for DP for PDmix. If server support dp in and out, should be removed
        if get_parallel_info_manager().is_distribution_enabled:
            return hidden_states

        dp = get_parallel_info_manager().get(ParallelType.ATTN_DP)
        if not dp.is_enabled():
            return hidden_states

        hidden_states = self._maybe_pad_cross_dp(hidden_states)

        # create output tensor
        gather_list = [torch.empty_like(hidden_states) for _ in range(dp.group_size)]

        # do all_gather
        torch.distributed.all_gather(gather_list, hidden_states, group=dp.process_group)
        output_parallel_ = torch.cat(gather_list, dim=0)
        hidden_states = self._maybe_unpad_cross_dp(output_parallel_)

        return hidden_states

    def _maybe_pad_cross_dp(self, hidden_states):
        # maybe pad hidden_states cross dp for all_gather
        forward_context = get_forward_context()

        num_tokens_across_dp_cpu = forward_context.dp_metadata.num_tokens_across_dp_cpu
        pad_size = max(num_tokens_across_dp_cpu) - hidden_states.shape[0]

        if pad_size > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_size))

        return hidden_states

    def _maybe_unpad_cross_dp(self, hidden_states):
        # unpad hidden_states after all_gather
        forward_context = get_forward_context()

        dp = get_parallel_info_manager().get(ParallelType.ATTN_DP)
        dp_world_size = dp.group_size
        num_tokens_across_dp_cpu = forward_context.dp_metadata.num_tokens_across_dp_cpu
        result = torch.empty(
            (num_tokens_across_dp_cpu.sum(), *hidden_states.shape[1:]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        hidden_states = hidden_states.view(dp.group_size, -1, *hidden_states.shape[1:])
        offset = 0
        for idx in range(dp_world_size):
            num_tokens_dp = num_tokens_across_dp_cpu[idx]
            result[offset : offset + num_tokens_dp] = hidden_states[idx, :num_tokens_dp]
            offset += num_tokens_dp
        hidden_states = result
        return hidden_states
