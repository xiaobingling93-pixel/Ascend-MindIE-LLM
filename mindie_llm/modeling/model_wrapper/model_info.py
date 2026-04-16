# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from typing import Any


class ModelInfo:
    device: Any
    dtype: Any
    data_byte_size: int
    num_layers: int
    num_kv_heads: int
    head_size: int
    k_head_size: int
    v_head_size: int
    enable_nz: bool
    index_head_dim: int
    num_index_heads: int

    def __init__(self, device, dtype, data_byte_size, num_layers, num_kv_heads, head_size, **kwargs):
        self.device = device
        self.dtype = dtype
        self.data_byte_size = data_byte_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.k_head_size = kwargs.get("k_head_size", self.head_size)
        self.v_head_size = kwargs.get("v_head_size", self.head_size)
        self.enable_nz = kwargs.get("enable_nz", False)
        self.kvcache_quant_layers = kwargs.get("kvcache_quant_layers", None)
        self.index_head_dim = kwargs.get("index_head_dim")
        self.num_index_heads = kwargs.get("num_index_heads")
