#!/usr/bin/env python
# coding=utf-8
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

from atb_llm.utils.data.quant_method_adapter import (
    QuantizationConfig,
)
from atb_llm.utils.quantize.pack_type import TransposeType
from atb_llm.utils.quantize.quant_type import LinearTypeV2, QuantType
from mindie_llm.runtime.layers.linear.linear import (
    RowParallelLinear as RowParallelLinearAdaptee,
    ColumnParallelLinear as ColumnParallelLinearAdaptee,
    MergedColumnParallelLinear as MergedColumnParallelLinearAdaptee,
    QKVParallelLinear as QKVParallelLinearAdaptee,
)
from mindie_llm.runtime.layers.embedding.embedding import (
    VocabParallelEmbedding as VocabParallelEmbeddingAdaptee,
    ParallelLMHead as ParallelLMHeadAdaptee,
)
from mindie_llm.runtime.layers.normalization import RMSNorm as RMSNormAdaptee
from mindie_llm.runtime.layers.quantization.unquantized import (
    UnquantizedEmbeddingMethod,
    UnquantizedNormMethod,
    UnquantizedLinearMethod,
)


class LayerSupportAtbGraph(ABC):
    _PLACEHOLDER = torch.tensor([1], dtype=torch.get_default_dtype(), device='npu')

    @abstractmethod
    def get_weights_for_atb_graph(self, padding: bool = True) -> list[torch.Tensor]:
        pass


class LinearLayerSupportAtbGraph(LayerSupportAtbGraph):
    @abstractmethod
    def get_weights_for_atb_graph(
        self, padding: bool = True,
        is_swiglu_quant_enabled: bool = False,
        quant_type: QuantType = None
    ) -> list[torch.Tensor]:
        pass

    @abstractmethod
    def get_linear_descs(self) -> list[LinearTypeV2]:
        pass

    @abstractmethod
    def get_weight_transpose_type(self) -> list[TransposeType]:
        pass


class RowParallelLinear(RowParallelLinearAdaptee, LayerSupportAtbGraph):
    def __init__(self, *args, **kwargs):
        kwargs["quant_config"] = QuantizationConfig(
            kwargs.get("quant_config"), UnquantizedLinearMethod)
        super().__init__(*args, **kwargs)

    def get_weights_for_atb_graph(
        self, padding: bool = True,
        is_swiglu_quant_enabled: bool = False,
        quant_type: QuantType = None
    ) -> list[torch.Tensor]:
        return self.quant_method.get_weights_for_atb_graph(
            self, padding=padding, is_swiglu_quant_enabled=is_swiglu_quant_enabled,
            quant_type=quant_type)

    def get_linear_descs(self) -> list[LinearTypeV2]:
        return [self.quant_method.get_linear_descs(self)]

    def get_weight_transpose_type(self) -> list[TransposeType]:
        return [self.quant_method.get_weight_transpose_type(self)]


class ColumnParallelLinear(ColumnParallelLinearAdaptee, LayerSupportAtbGraph):
    def __init__(self, *args, **kwargs):
        kwargs["quant_config"] = QuantizationConfig(
            kwargs.get("quant_config"), UnquantizedLinearMethod)
        super().__init__(*args, **kwargs)

    def get_weights_for_atb_graph(
        self, padding: bool = True,
        is_swiglu_quant_enabled: bool = False,
        quant_type: QuantType = None
    ) -> list[torch.Tensor]:
        if is_swiglu_quant_enabled:
            raise ValueError("Cannot set `is_swiglu_quant_enabled` to True in `ColumnParallelLinear`.")
        return self.quant_method.get_weights_for_atb_graph(self, padding=padding, quant_type=quant_type)

    def get_linear_descs(self) -> list[LinearTypeV2]:
        return [self.quant_method.get_linear_descs(self)]

    def get_weight_transpose_type(self) -> list[TransposeType]:
        return [self.quant_method.get_weight_transpose_type(self)]


class MergedColumnParallelLinear(MergedColumnParallelLinearAdaptee, ColumnParallelLinear):
    def __init__(self, *args, **kwargs):
        # Set `sub_layer_cls` to the adapted version of ColumnParallelLinear to support atbgraph
        MergedColumnParallelLinearAdaptee.sub_layer_cls = ColumnParallelLinear
        super().__init__(*args, **kwargs)

    def get_weights_for_atb_graph(
        self, padding: bool = True,
        is_swiglu_quant_enabled: bool = False,
        quant_type: QuantType = None
    ) -> list[torch.Tensor]:
        if is_swiglu_quant_enabled:
            raise ValueError("Cannot set `is_swiglu_quant_enabled` to True in `MergedColumnParallelLinear`.")
        # When MergedColumnParallelLinear has multiple linear_modules (different quant types per sub-layer),
        # collect weights from each module
        if len(self.linear_modules) > 1:
            weights = []
            for module in self.linear_modules:
                weights_per_module = module.get_weights_for_atb_graph(padding=padding, quant_type=quant_type)
                weights.extend(weights_per_module)
        else:
            # Single merged linear layer: use parent implementation and append placeholders
            weights = super().get_weights_for_atb_graph(padding=padding, quant_type=quant_type)
            weights.extend([self._PLACEHOLDER] * 6)
        # Ensure total weight count does not exceed the limit (12)
        if len(weights) > 12:
            raise ValueError("Number of weights from `MergedColumnParallelLinear` should not exceed 12.")
        return weights

    def get_linear_descs(self) -> list[LinearTypeV2]:
        # When MergedColumnParallelLinear has multiple linear_modules (different quant types per sub-layer),
        # collect linear descs from each module
        if len(self.linear_modules) > 1:
            linear_descs = []
            for module in self.linear_modules:
                linear_descs.extend(module.get_linear_descs())
        else:
            # When MergedColumnParallelLinear is a merged linear layer,
            # use parent implementation and append INVALID marker
            linear_descs = super().get_linear_descs()
            linear_descs.append(LinearTypeV2.INVALID)
        return linear_descs

    def get_weight_transpose_type(self) -> list[TransposeType]:
        # When multiple linear_modules exist, collect transpose types from each module and concatenate
        if len(self.linear_modules) > 1:
            transpose_type = []
            for module in self.linear_modules:
                transpose_type.extend(module.get_weight_transpose_type())
        else:
            # When MergedColumnParallelLinear is a merged linear layer,
            # use parent implementation and append INVALID marker
            transpose_type = super().get_weight_transpose_type()
            transpose_type.append(TransposeType.INVALID)
        return transpose_type


class QKVParallelLinear(QKVParallelLinearAdaptee, ColumnParallelLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_weights_for_atb_graph(
        self, padding: bool = True,
        is_swiglu_quant_enabled: bool = False,
        quant_type: QuantType = None
    ) -> list[torch.Tensor]:
        if is_swiglu_quant_enabled:
            raise ValueError("Cannot set `is_swiglu_quant_enabled` to True in `QKVParallelLinear`.")
        weights = self.quant_method.get_weights_for_atb_graph(self, padding=padding, quant_type=quant_type)
        weights.extend([self._PLACEHOLDER] * 12)
        return weights

    def get_linear_descs(self) -> list[LinearTypeV2]:
        linear_descs = super().get_linear_descs()
        linear_descs.extend([LinearTypeV2.INVALID] * 2)
        return linear_descs

    def get_weight_transpose_type(self) -> list[TransposeType]:
        transpose_type = super().get_weight_transpose_type()
        transpose_type.extend([TransposeType.INVALID] * 2)
        return transpose_type


class VocabParallelEmbedding(VocabParallelEmbeddingAdaptee, LayerSupportAtbGraph):
    def __init__(self, *args, **kwargs):
        kwargs["quant_config"] = QuantizationConfig(
            kwargs.get("quant_config"), UnquantizedEmbeddingMethod)
        super().__init__(*args, **kwargs)

    def get_weights_for_atb_graph(self, padding: bool = True):
        weights = self.quant_method.get_weights_for_atb_graph(self, padding=padding)
        return weights


class ParallelLMHead(ParallelLMHeadAdaptee, LayerSupportAtbGraph):
    def __init__(self, *args, **kwargs):
        kwargs["quant_config"] = QuantizationConfig(
            kwargs.get("quant_config"), UnquantizedLinearMethod)
        super().__init__(*args, **kwargs)

    def get_weights_for_atb_graph(
        self, padding: bool = True,
        is_swiglu_quant_enabled: bool = False,
        quant_type: QuantType = None
    ) -> list[torch.Tensor]:
        weights = self.quant_method.get_weights_for_atb_graph(self, padding=padding)
        return weights

    def get_linear_descs(self) -> list[LinearTypeV2]:
        return [self.quant_method.get_linear_descs(self)]

    def get_weight_transpose_type(self) -> list[TransposeType]:
        return [self.quant_method.get_weight_transpose_type(self)]


class RMSNorm(RMSNormAdaptee, LayerSupportAtbGraph):
    def __init__(self, *args, **kwargs):
        kwargs["quant_config"] = QuantizationConfig(
            kwargs.get("quant_config"), UnquantizedNormMethod)
        super().__init__(*args, **kwargs)

    def get_weights_for_atb_graph(self, padding: bool = True) -> list[torch.Tensor]:
        return self.quant_method.get_weights_for_atb_graph(self, padding=padding)
