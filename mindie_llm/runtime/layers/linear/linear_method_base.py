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

from mindie_llm.runtime.layers.quantization.quantization_method_base import QuantizationMethodBase


class LinearMethodBase(QuantizationMethodBase):
    @abstractmethod
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        bias: bool,
        weight_dtype: torch.dtype,
        bias_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Creates weights for the linear layer.

        Args:
            layer: The linear layer to be processed.
            input_size_per_partition: Input dimension size per partition (for model parallelism).
            output_partition_sizes: List of output dimension sizes per partition.
            bias: Whether to include bias in the layer.
            weight_dtype: Data type for weight tensors.
            bias_dtype: Data type for bias tensors.
            extra_weight_attrs: Additional attributes for weight initialization.

        Note:
            Must be implemented by subclasses. Ensure partition sizes match layer configuration.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies the quantized linear layer to input tensor.
        Args:
            layer: The quantized linear layer (must be initialized via create_weights).
            x: Input tensor to process.
        Returns:
            Output tensor after applying the layer.
        Note:
            Layer must be initialized with create_weights; undefined behavior otherwise.
        """
        raise NotImplementedError
