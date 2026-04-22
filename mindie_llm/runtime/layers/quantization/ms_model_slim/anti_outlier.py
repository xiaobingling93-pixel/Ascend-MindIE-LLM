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

from mindie_llm.runtime.layers.quantization.quantization_method_base import QuantizationMethodBase
from mindie_llm.runtime.layers.parameter import BaseParameter


class AntiOutlierNormMethod(QuantizationMethodBase):
    """
    Quantization method for Anti-Outlier Normalization.
    Uses NPU-optimized RMS Norm with optional residual connection and bias addition.
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        hidden_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """
        Args:
            layer: The layer instance to register weights to
            hidden_size: Dimension of the hidden state
            params_dtype: Data type for parameters (weight and bias)
            extra_weight_attrs: Extra attributes to add to parameters
        """
        weight = BaseParameter(torch.ones(hidden_size, dtype=params_dtype))
        weight.add_attrs(extra_weight_attrs)
        layer.register_parameter("weight", weight)

        bias = BaseParameter(torch.zeros(hidden_size, dtype=params_dtype))
        bias.add_attrs(extra_weight_attrs)
        layer.register_parameter("bias", bias)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Applies Anti-Outlier Normalization to the input tensor.

        Note:
            If residual is provided, uses fused `npu_add_rms_norm` for performance.
            Bias is added after normalization.

        Returns:
            - If residual is None: Normalized tensor.
            - If residual is provided: Tuple of (normalized_tensor, residual_tensor).
        """
        if residual is not None:
            x, _, residual = torch_npu.npu_add_rms_norm(x, residual, layer.weight.data, layer.variance_epsilon)
            x = x + layer.bias.data
            return x, residual

        x, _ = torch_npu.npu_rms_norm(x, layer.weight.data, layer.variance_epsilon)
        x = x + layer.bias.data
        return x
