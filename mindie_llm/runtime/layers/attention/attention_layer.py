# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from abc import ABC, abstractmethod

import torch

from mindie_llm.runtime.layers.custom_layer import CustomLayer
from mindie_llm.runtime.layers.quantization.quantization_config_base import QuantizationConfigBase
from .backend import get_attn_backend
from .backend.abstract import AttentionBackend
from . import update_global_attn_dict


class AttentionLayerBase(CustomLayer, ABC):
    @abstractmethod
    def get_attn_backend(self) -> type[AttentionBackend]:
        pass


class AttentionQuant(torch.nn.Module):
    def __init__(self, head_size, num_kv_heads, num_kv_heads_replicas, weight_dtype, quant_config, prefix):
        super().__init__()
        self.prefix = None  # prefix differs from different quant methods, e.g., k_proj for c8, fa_q for fa3
        self.quant_method = quant_config.get_quant_method(self, prefix=prefix) if quant_config else None
        self.enable_kv_quant = False
        if self.quant_method:
            self.quant_method.create_weights(
                self,
                head_size=head_size,
                num_kv_heads=num_kv_heads,
                num_kv_heads_replicas=num_kv_heads_replicas,
                weight_dtype=weight_dtype,
                prefix=prefix,
            )


class Attention(AttentionLayerBase):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        num_kv_heads_replicas: int | None = None,
        weight_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfigBase | None = None,
        prefix: str = "",
        attn_backend: type[AttentionBackend] | None = None,
        **extra_impl_args,
    ) -> None:
        super().__init__()
        self.key_cache = None
        self.value_cache = None
        update_global_attn_dict(prefix, self)
        self.prefix = prefix

        if attn_backend is None:
            self.attn_backend = get_attn_backend(use_mla=False)
        else:
            self.attn_backend = attn_backend
        self.attn_quant = AttentionQuant(
            head_size, num_kv_heads, num_kv_heads_replicas, weight_dtype, quant_config, prefix
        )

        impl_cls = self.attn_backend.get_impl_cls()
        self.impl = impl_cls(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            self.attn_quant.quant_method,
            **extra_impl_args,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        self_kv_cache = (self.key_cache, self.value_cache)
        return self.impl.forward(self, query, key, value, self_kv_cache)
