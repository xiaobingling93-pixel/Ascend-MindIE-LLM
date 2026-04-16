# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import torch
import torch_npu
import numpy as np


from mindie_llm.runtime.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from mindie_llm.runtime.layers.parameter import (
    ModelWeightParameter,
)
from mindie_llm.runtime.utils.distributed.utils import even_divide


class W4A8PerTokenFusedMoEMethod(FusedMoEMethodBase):
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        **extra_weight_attrs,
    ):
        gate_up_weight = ModelWeightParameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                hidden_size,
                dtype=torch.int8,
            ),
        )
        gate_up_weight.add_attrs({self.INPUT_DIM: 1, self.OUTPUT_DIM: 0, **extra_weight_attrs})
        layer.register_parameter("gate_up_weight", gate_up_weight)

        down_weight = ModelWeightParameter(
            torch.empty(
                num_experts,
                even_divide(hidden_size, 2),
                intermediate_size_per_partition,
                dtype=torch.int8,
            ),
        )
        down_weight.add_attrs({self.INPUT_DIM: 1, self.OUTPUT_DIM: 0, **extra_weight_attrs})
        layer.register_parameter("down_weight", down_weight)
        layer.output_dtype = torch.float16

        gate_up_weight_scale = ModelWeightParameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                1,
                dtype=torch.float32,
            ),
        )
        gate_up_weight_scale.add_attrs({self.OUTPUT_DIM: 0, **extra_weight_attrs})
        layer.register_parameter("gate_up_weight_scale", gate_up_weight_scale)

        down_weight_scale = ModelWeightParameter(
            torch.empty(
                num_experts,
                hidden_size,
                1,
                dtype=torch.float32,
            ),
        )
        down_weight_scale.add_attrs({self.INPUT_DIM: 0, **extra_weight_attrs})
        layer.register_parameter("down_weight_scale", down_weight_scale)
        gate_up_scale_bias = ModelWeightParameter(
            torch.empty(num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32)
        )
        gate_up_scale_bias.add_attrs({self.OUTPUT_DIM: 0, **extra_weight_attrs})
        layer.register_parameter("gate_up_scale_bias", gate_up_scale_bias)
        down_scale_bias = ModelWeightParameter(
            torch.empty(num_experts, hidden_size, even_divide(16, layer.moe_tp_size), dtype=torch.float32)
        )
        down_scale_bias.add_attrs({self.INPUT_DIM: 1, self.OUTPUT_DIM: 0, **extra_weight_attrs})

        layer.register_parameter("down_scale_bias", down_scale_bias)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        group_list: torch.Tensor,
        group_list_type: int = 1,
        dynamic_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        if dynamic_scale is None:
            x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
        else:
            pertoken_scale = dynamic_scale

        hidden_states = torch_npu.npu_grouped_matmul(
            x=[x],
            weight=[layer.gate_up_weight],
            scale=[layer.gate_up_weight_scale],
            bias=[layer.gate_up_scale_bias],
            per_token_scale=[pertoken_scale],
            split_item=2,  # 切分模式的说明 2表示输出需要切分
            group_list_type=group_list_type,
            group_type=0,  # 代表需要分组的轴 0代表m轴分组
            group_list=group_list,
            output_dtype=layer.output_dtype,
        )[0]

        hidden_states = torch_npu.npu_swiglu(hidden_states)
        hidden_states, swiglu_out_scale = torch_npu.npu_dynamic_quant(hidden_states)

        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.down_weight],
            scale=[layer.down_weight_scale],
            bias=[layer.down_scale_bias],
            per_token_scale=[swiglu_out_scale],
            split_item=2,  # 切分模式的说明 2表示输出需要切分
            group_list_type=group_list_type,
            group_type=0,  # 代表需要分组的轴 0代表m轴分组
            group_list=group_list,
            output_dtype=layer.output_dtype,
        )[0]
        return hidden_states

    def process_scale(self, weight: torch.Tensor, scale):
        scale_np = scale.cpu().numpy()
        scale_np.dtype = np.uint32
        scale_uint64_tensor = torch.from_numpy(scale_np.astype(np.int64)).npu()
        return scale_uint64_tensor

    def pack_to_int32(self, weight: torch.Tensor):
        # pack 4 int8(int4*2) to int32, because in pytorch, we need to use int32 to represent int4
        return weight.view(torch.int32).contiguous()

    def process_weights_after_loading(self, layer):
        layer.gate_up_weight.data = layer.gate_up_weight.data.transpose(-2, -1).contiguous()
        layer.gate_up_weight.data = torch_npu.npu_format_cast(layer.gate_up_weight.data, 29)
        layer.down_weight.data = layer.down_weight.data.transpose(-2, -1).contiguous()
        layer.down_weight.data = torch_npu.npu_format_cast(layer.down_weight.data, 29)
        layer.gate_up_weight_scale.data = layer.gate_up_weight_scale.data.view(
            layer.gate_up_weight_scale.data.shape[0], -1
        )
        layer.gate_up_weight_scale.data = self.process_scale(layer.gate_up_weight, layer.gate_up_weight_scale.data)
        layer.gate_up_weight_scale.data = layer.gate_up_weight_scale.data.unsqueeze(1)
        layer.down_weight_scale.data = self.process_scale(layer.down_weight, layer.down_weight_scale.data)
        layer.down_weight_scale.data = layer.down_weight_scale.data.unsqueeze(1)
        layer.gate_up_scale_bias.data = layer.gate_up_scale_bias.data.transpose(1, 2).contiguous().sum(axis=1)
        layer.down_scale_bias.data = layer.down_scale_bias.data.transpose(1, 2).contiguous().sum(axis=1)
        layer.gate_up_weight.data = self.pack_to_int32(layer.gate_up_weight.data)
        layer.down_weight.data = self.pack_to_int32(layer.down_weight.data)
