#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Implement part of this file based on vllm-ascend.
#
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import List, Optional, Union, Type, Dict

import weakref
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parameter import Parameter

from mindie_llm.runtime.utils.distributed import get_parallel_info_manager
from mindie_llm.runtime.utils.distributed.parallel_info_manager import ParallelInfo, ParallelType
from mindie_llm.runtime.model_runner.forward_context_exp import get_forward_context
from mindie_llm.utils.env import ENV as ENV_utils

# Mapping from Linear layer class name to candidate custom LinearOp implementations.
# Each value is a callable returning a tuple of op classes to try in order.
# (Using a lambda delays evaluation so the op classes can be defined below.)
_LINEAR_TO_CUSTOM_OP_DISPATCH_TABLE = {
    "RowParallelLinear": lambda: (SequenceRowParallelOp,),
    "ColumnParallelLinear": lambda: (SequenceColumnParallelOp,),
}


# Cache for LinearOp instances, keyed by the layer object and the selected op class:
#   layer -> { op_cls -> op_instance }
# WeakKeyDictionary is used so that once a layer is garbage-collected,
# its cache entry is automatically removed (prevents memory leaks).
_LINEAR_OP_INSTANCE_CACHE: "weakref.WeakKeyDictionary[object, Dict[Type[LinearOp], LinearOp]]" = (
    weakref.WeakKeyDictionary()
)


class LinearOp:
    """Base class for different linear forward methods."""

    def __init__(self, layer):
        self.layer = layer
        self.bias = None
        self.skip_bias_add = None
        self.return_bias = None
        self.quant_method = None
        self.prefix = None
        self.parallel_info = None

    def __call__(self, input_):
        return self.forward(input_)

    @staticmethod
    def enable(layer_cls) -> bool:
        # Whether this LinearOp should be used for the given layer under current context like forward_context.
        # Subclasses should override this.
        return False

    # Call this to update attrs by self.layer before calling the __call__ method of Linear.
    def update_attrs(self):
        if hasattr(self.layer, "bias"):
            self.bias = self.layer.bias
        self.skip_bias_add = self.layer.skip_bias_add
        self.return_bias = self.layer.return_bias
        self.quant_method = self.layer.quant_method
        self.prefix = self.layer.prefix
        self.parallel_info = self.layer.parallel_info

        if self.quant_method is None:
            raise AttributeError("The quant_method is None.")

    def apply_impl(self, input_):
        raise NotImplementedError

    # Replace the __call__ of LinearBase to customize the layer process on RUNTIME.
    def forward(self, input_):
        output, output_bias = self.apply_impl(input_)
        if not self.return_bias:
            return output
        return output, output_bias


class ColumnParallelOp(LinearOp):
    def __init__(self, layer):
        super().__init__(layer)
        self.gather_output = None

    def update_attrs(self):
        super().update_attrs()
        self.gather_output = self.layer.gather_output
        if self.gather_output:
            raise NotImplementedError("`ColumnParallelOp` doesn't support setting `gather_output` to True.")


class RowParallelOp(LinearOp):
    def __init__(self, layer):
        super().__init__(layer)
        self.reduce_results = None

    def update_attrs(self):
        super().update_attrs()
        self.reduce_results = self.layer.reduce_results


class SequenceColumnParallelOp(ColumnParallelOp):
    @staticmethod
    def enable(layer_cls) -> bool:
        forward_context = get_forward_context()
        if not forward_context.batch_descriptor.is_flash_comm_enabled:
            return False
        # AllGather should be handled elsewhere in MLA.
        if "q_b_proj" in layer_cls.prefix:
            return False
        return True

    def apply_impl(self, input_: torch.Tensor) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        """Linear layer with column parallelism."""
        input_ = maybe_all_gather_and_maybe_unpad(input_, self.parallel_info)
        output_parallel = self.quant_method.apply(self.layer, input_)

        output_bias = self.bias if self.skip_bias_add else None
        return output_parallel, output_bias


class SequenceRowParallelOp(RowParallelOp):
    @staticmethod
    def enable(layer_cls) -> bool:
        forward_context = get_forward_context()
        if forward_context.batch_descriptor.is_flash_comm_enabled:
            return True
        return False

    def apply_impl(self, input_: torch.Tensor) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        """Linear layer with row parallelism."""
        output = self.quant_method.apply(self.layer, input_)

        if self.parallel_info.group_size > 1 and self.reduce_results:
            reduce_scatter_output = maybe_pad_and_reduce_scatter(output, self.parallel_info)
            output = reduce_scatter_output

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


def maybe_pad_and_reduce_scatter(
    input_: torch.Tensor,
    parallel_info: ParallelInfo,
) -> torch.Tensor:
    world_size = parallel_info.group_size

    # prepare pad_size
    pad_size = (world_size - (input_.shape[0] % world_size)) % world_size

    if pad_size > 0:
        input_ = F.pad(input_, (0, 0, 0, pad_size))

    reduce_scatter_output = torch.empty(
        (input_.shape[0] // world_size, *input_.shape[1:]), dtype=input_.dtype, device=input_.device
    )
    dist.reduce_scatter_tensor(reduce_scatter_output, input_, group=parallel_info.process_group)
    return reduce_scatter_output


def maybe_all_gather_and_maybe_unpad(
    input_: torch.Tensor,
    parallel_info: ParallelInfo,
) -> torch.Tensor:
    # prepare
    comm_group = parallel_info.process_group
    world_size = parallel_info.group_size

    if world_size <= 1:
        return input_

    # create output tensor
    gather_list: List[torch.Tensor] = [torch.empty_like(input_) for _ in range(world_size)]

    # do all_gather
    dist.all_gather(gather_list, input_, group=comm_group)
    output_parallel_ = torch.cat(gather_list, dim=0)

    # unpad
    unpad_input_ = maybe_unpad_cross_dp(output_parallel_, parallel_info)
    return unpad_input_


def maybe_unpad_cross_dp(input_: torch.Tensor, parallel_info: ParallelInfo):
    """Auto unpadding the dp data for based on current Allgather worldsize."""
    attn_tp = get_parallel_info_manager().get(ParallelType.ATTN_TP)
    attn_dp = get_parallel_info_manager().get(ParallelType.ATTN_DP)
    attn_tp_world_size = attn_tp.group_size
    cur_world_size = parallel_info.group_size
    if cur_world_size < attn_tp_world_size or cur_world_size % attn_tp_world_size != 0:
        raise NotImplementedError(
            "world_size of AllGather must be greater than or equal to world_size "
            "of ATTN_TP, i.e., it must be a multiple of attn_tp_world_size."
        )

    try:
        forward_context = get_forward_context()
    except AssertionError:
        return input_

    if ENV_utils.model_runner_exp:
        num_tokens_across_dp_cpu = forward_context.dp_metadata.num_tokens_across_dp_cpu
    else:
        num_tokens_across_dp_cpu = forward_context.num_tokens_across_dp_cpu

    ratio = cur_world_size // attn_tp_world_size
    cur_dp_num_tokens_across_dp_cpu = num_tokens_across_dp_cpu.view(-1, ratio)[attn_dp.rank // ratio]
    cur_dp_size = len(cur_dp_num_tokens_across_dp_cpu)

    result = torch.empty(
        (cur_dp_num_tokens_across_dp_cpu.sum(), *input_.shape[1:]), device=input_.device, dtype=input_.dtype
    )
    input_ = input_.view(cur_dp_size, -1, *input_.shape[1:])
    offset = 0
    for idx in range(cur_dp_size):
        num_tokens_dp = cur_dp_num_tokens_across_dp_cpu[idx]
        result[offset : offset + num_tokens_dp] = input_[idx, :num_tokens_dp]
        offset += num_tokens_dp
    input_ = result
    return result


def get_linear_custom_op(layer) -> LinearOp | None:
    table = _LINEAR_TO_CUSTOM_OP_DISPATCH_TABLE

    # Select the best matching LinearOp class for this layer under the 'enable' method of LinearOp.
    # We walk the MRO so subclasses can reuse parent dispatch rules.
    selected_op_cls = None
    for cls in layer.__class__.__mro__[:-1]:  # pass Object
        cls_name = cls.__name__
        if cls_name not in table:
            continue
        for op_cls in table[cls_name]():
            if op_cls.enable(layer):
                selected_op_cls = op_cls
                break
        if selected_op_cls is not None:
            break

    if selected_op_cls is None:
        return None

    # Fetch or create the per-layer cache bucket.
    per_layer_cache = _LINEAR_OP_INSTANCE_CACHE.get(layer)
    if per_layer_cache is None:
        per_layer_cache = {}
        _LINEAR_OP_INSTANCE_CACHE[layer] = per_layer_cache

    # Fetch or create the cached op instance for (layer, selected_op_cls).
    # This avoids allocating a new op object on every forward.
    op = per_layer_cache.get(selected_op_cls)
    if op is None:
        op = selected_op_cls(layer)
        per_layer_cache[selected_op_cls] = op

    return op
