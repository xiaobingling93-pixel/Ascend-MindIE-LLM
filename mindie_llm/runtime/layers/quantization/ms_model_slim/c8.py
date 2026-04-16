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
from torch import nn
import numpy as np
from mindie_llm.runtime.layers.quantization.quantization_method_base import QuantizationMethodBase
from mindie_llm.runtime.layers.parameter import ColumnParameter
from mindie_llm.runtime.utils.distributed import get_parallel_info_manager


class AttnQuantMethodBase(QuantizationMethodBase):
    def __init__(self):
        super().__init__()
        self.k_quant_offset = None
        self.v_quant_offset = None
        self.k_quant_scale = None
        self.v_quant_scale = None
        self.kv_dequant_scale = None
        self.kv_dequant_offset = None


class KVQuantMethod(AttnQuantMethodBase):
    def __init__(self):
        super().__init__()
        self.num_kv_heads_replicas = 0
        self.kv_size = None

    def create_weights(
        self,
        layer: nn.Module,
        head_size: int,
        num_kv_heads: int,
        num_kv_heads_replicas: int,
        weight_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.kv_size = num_kv_heads * head_size
        self.num_kv_heads_replicas = num_kv_heads_replicas
        scale = ColumnParameter(data=torch.empty(self.kv_size * 2, dtype=weight_dtype))
        offset = ColumnParameter(data=torch.empty(self.kv_size * 2, dtype=weight_dtype))
        scale.add_attrs(
            {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            }
        )
        offset.add_attrs(
            {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            }
        )
        prefix = extra_weight_attrs.get("prefix", "")
        layer.prefix = [f"{prefix}.k_proj", f"{prefix}.v_proj"]
        layer.register_parameter("kv_cache_scale", scale)
        layer.register_parameter("kv_cache_offset", offset)

    def weight_loader(self, param: ColumnParameter, loaded_weight: torch.Tensor, loaded_shard_id: int) -> None:
        shard_offset = self.kv_size * loaded_shard_id
        shard_size = self.kv_size
        shard_id = get_parallel_info_manager().rank // self.num_kv_heads_replicas  # kv head replicas index for tp
        param.load_merged_column_weight(loaded_weight, shard_id, shard_offset, shard_size)

    def apply(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        key_int8 = torch_npu.npu_quantize(key, self.k_quant_scale, self.k_quant_offset, torch.qint8, -1, False)
        value_int8 = torch_npu.npu_quantize(value, self.v_quant_scale, self.v_quant_offset, torch.qint8, -1, False)
        return (key_int8, value_int8)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        kv_cache_scale = layer.kv_cache_scale
        kv_cache_offset = layer.kv_cache_offset
        k_scale, v_scale = torch.split(kv_cache_scale, [self.kv_size, self.kv_size], 0)
        k_offset, v_offset = torch.split(kv_cache_offset, [self.kv_size, self.kv_size], 0)
        # FP16和BF16都先cast成FP32，按位转成INT32
        device = k_scale.device
        k_offset = torch.from_numpy(
            np.frombuffer(k_offset.to(torch.float32).cpu().numpy().tobytes(), dtype=np.int32).copy()
        ).to(device)
        v_offset = torch.from_numpy(
            np.frombuffer(v_offset.to(torch.float32).cpu().numpy().tobytes(), dtype=np.int32).copy()
        ).to(device)
        dtype = k_scale.dtype
        self.k_quant_offset = nn.Parameter(k_offset.to(dtype), requires_grad=False)
        self.v_quant_offset = nn.Parameter(v_offset.to(dtype), requires_grad=False)
        self.k_quant_scale = nn.Parameter(k_scale.reciprocal(), requires_grad=False)
        self.v_quant_scale = nn.Parameter(v_scale.reciprocal(), requires_grad=False)
        self.kv_dequant_scale = nn.Parameter(kv_cache_scale.data.view(2, -1), requires_grad=False)
        self.kv_dequant_offset = nn.Parameter(kv_cache_offset.data.view(2, -1), requires_grad=False)
