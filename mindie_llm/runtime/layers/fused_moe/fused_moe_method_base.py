# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
#
# Implement part of this file based on vllm-project/vllm
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


class FusedMoEMethodBase(QuantizationMethodBase):
    """Base class for different (maybe quantized) fused moe methods."""

    @abstractmethod
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        weight_dtype: torch.dtype,
        bias_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        group_list: torch.Tensor,
        group_list_type: int = 1,
        dynamic_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        raise NotImplementedError
