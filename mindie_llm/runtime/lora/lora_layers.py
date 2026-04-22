# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import math
from abc import ABC, abstractmethod
from typing import List

import torch
import torch_npu
import torch.nn.functional as F
from torch.nn import Parameter

from mindie_llm.runtime.utils.npu.device_utils import get_npu_node_info, DeviceType
from mindie_llm.runtime.layers.custom_layer import CustomLayer
from mindie_llm.runtime.config.mindie_llm_config import LoraModelConfig
from mindie_llm.runtime.layers.linear.linear import LinearBase, ColumnParallelLinear, RowParallelLinear


class BaseLayerWithLoRA(CustomLayer, ABC):
    def __init__(self):
        super().__init__()

    @classmethod
    def can_replace_layer(cls, source_layer: CustomLayer) -> bool:
        """Return True if the layer can be replaced by this LoRA layer."""
        raise NotImplementedError

    @abstractmethod
    def create_lora_weights(self, lora_model_config: LoraModelConfig, dtype, device):
        """Initialize lora weights stack"""
        ...

    @abstractmethod
    def slice_lora_a(self, lora_tensors_dic: dict[str, torch.Tensor], prefixes: List[str]) -> torch.Tensor:
        """Slice lora a weight for TP and return lora a weight"""
        ...

    @abstractmethod
    def slice_lora_b(
        self, lora_tensors_dic: dict[str, torch.Tensor], prefixes: List[str], scales: List[int]
    ) -> torch.Tensor:
        """Slice lora b weight for TP and return lora b weight"""
        ...

    @abstractmethod
    def set_lora(self, index: int, lora_a: torch.Tensor, lora_b: torch.Tensor):
        """Overwrite lora weight at index"""
        ...

    @abstractmethod
    def reset_lora(self, index: int):
        """Clear lora weight at index"""
        ...


class ParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(self, base_layer: LinearBase):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer_prefixes = (
            [self.base_layer.prefix] if isinstance(self.base_layer.prefix, str) else self.base_layer.prefix
        )
        self.pack_num = len(self.base_layer_prefixes)
        self.base_weight_shape = self.get_base_weight_shape(base_layer)
        self.need_nz = False
        self.tp_rank = self.base_layer.tp_rank
        self.base_input_size_per_partition = self.base_layer.input_size_per_partition
        self.base_output_partition_sizes = self.base_layer.output_partition_sizes
        self.lora_a_stacked = Parameter(torch.tensor([]))
        self.lora_b_stacked = Parameter(torch.tensor([]))

    @staticmethod
    def get_base_weight_shape(base_layer: LinearBase):
        k, n = base_layer.weight.shape
        return n, k

    @staticmethod
    def weight_format_cast(tensor: torch.Tensor):
        """Transdata to Fractal_NZ format"""
        torch_npu.npu_format_cast_(tensor, 29)
        return tensor

    @staticmethod
    def get_padding_size(x: int, need_nz: bool = False):
        """Align up num to 16 or 64 according the soc version"""
        if need_nz:
            padding_size = math.ceil(x / 16) * 16  # Align up 16
        else:
            padding_size = math.ceil(x / 64) * 64  # Align up 64
        return padding_size

    @staticmethod
    def padding_tensor(padding_size, tensor):
        """Apply padding to a tensor along its first dimension"""
        padding_shape = [0, 0] * len(tensor.shape)
        padding_shape[-1] = (padding_size - tensor.shape[0] % padding_size) % padding_size
        return F.pad(tensor, padding_shape)

    def create_lora_weights(self, lora_model_config: LoraModelConfig, dtype, device):
        self.dtype = dtype
        self.device = device
        if self.dtype not in [torch.float16, torch.bfloat16]:
            raise RuntimeError("LoRA only supports float16 and bfloat16 currently.")
        if get_npu_node_info().get_device_type() == DeviceType.ASCEND_310P:
            self.need_nz = True
        max_loras = lora_model_config.max_loras
        max_lora_rank = lora_model_config.max_lora_rank
        padding_size = self.get_padding_size(max_lora_rank * self.pack_num, self.need_nz)
        n, k = self.base_weight_shape
        # GroupedMatMulOperation and LinearOperation need lora_A of shape [r, k] and lora_B of shape [r, n]
        lora_a = torch.zeros(max_loras + 1, padding_size, k, dtype=self.dtype, device=self.device)  # Num of LoRA, r, k
        lora_b = torch.zeros(max_loras + 1, padding_size, n, dtype=self.dtype, device=self.device)  # Num of LoRA, r, n
        self.lora_a_stacked.data = self.weight_format_cast(lora_a) if self.need_nz else lora_a
        self.lora_b_stacked.data = self.weight_format_cast(lora_b) if self.need_nz else lora_b

    def set_lora(self, index: int, lora_a: torch.Tensor, lora_b: torch.Tensor):
        dim_r = lora_a.shape[0]
        padding_size = self.get_padding_size(dim_r, self.need_nz)
        lora_a = self.padding_tensor(padding_size, lora_a).to(self.device)
        lora_b = self.padding_tensor(padding_size, lora_b).to(self.device)
        if self.need_nz:
            lora_a = self.weight_format_cast(lora_a)
            lora_b = self.weight_format_cast(lora_b)
        self.lora_a_stacked.data[index, :padding_size].copy_(lora_a)
        self.lora_b_stacked.data[index, :padding_size].copy_(lora_b)

    def reset_lora(self, index: int):
        self.lora_a_stacked.data[index].zero_()
        self.lora_b_stacked.data[index].zero_()


class ColumnParallelLinearWithLoRA(ParallelLinearWithLoRA):
    def __init__(self, base_layer: ColumnParallelLinear):
        super().__init__(base_layer)

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: CustomLayer,
    ) -> bool:
        return isinstance(source_layer, ColumnParallelLinear)

    def slice_lora_a(self, lora_tensors_dic: dict[str, torch.Tensor], prefixes: List[str]) -> torch.Tensor:
        # not TP on LoRA_A weight
        weight_tensors = [lora_tensors_dic[f"{p}.weight"] for p in prefixes]
        if len(weight_tensors) > 1:
            weight_tensors = [torch.cat(weight_tensors)]
        return weight_tensors[0].to(self.dtype)

    def slice_lora_b(
        self, lora_tensors_dic: dict[str, torch.Tensor], prefixes: List[str], scales: List[int]
    ) -> torch.Tensor:
        # only perform n-dim TP on LoRA_B weight
        weight_tensors = []
        for p, partition_size in zip(prefixes, self.base_output_partition_sizes):
            start_idx = self.tp_rank * partition_size
            end_idx = (self.tp_rank + 1) * partition_size
            weight_tensors.append(lora_tensors_dic[f"{p}.weight"][start_idx:end_idx, :])

        if len(weight_tensors) == 1:
            weight_tensors = [(weight_tensors[0] * scales[0])]
        if len(weight_tensors) > 1:
            scaled_lora_weights = []
            for weight_tensor, scale in zip(weight_tensors, scales):
                scaled_lora_weights.append(weight_tensor * scale)
            weight_tensors = [torch.block_diag(*scaled_lora_weights)]
        weight_tensors = weight_tensors[0].T.contiguous()
        return weight_tensors.to(self.dtype)


class RowParallelLinearWithLoRA(ParallelLinearWithLoRA):
    def __init__(self, base_layer: RowParallelLinear):
        super().__init__(base_layer)

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: CustomLayer,
    ) -> bool:
        return isinstance(source_layer, RowParallelLinear)

    def slice_lora_a(self, lora_tensors_dic: dict[str, torch.Tensor], prefixes: List[str]) -> torch.Tensor:
        # only perform n-dim TP on LoRA_A weight
        weight_tensors = [lora_tensors_dic[f"{p}.weight"] for p in prefixes]
        start_idx = self.tp_rank * self.base_input_size_per_partition
        end_idx = (self.tp_rank + 1) * self.base_input_size_per_partition
        weight_tensors = weight_tensors[0][:, start_idx:end_idx]
        return weight_tensors.to(self.dtype)

    def slice_lora_b(
        self, lora_tensors_dic: dict[str, torch.Tensor], prefixes: List[str], scales: List[int]
    ) -> torch.Tensor:
        # not TP on LoRA_B weight
        weight_tensors = [lora_tensors_dic[f"{p}.weight"] for p in prefixes]
        weight_tensors = weight_tensors[0] * scales[0]
        weight_tensors = weight_tensors.T.contiguous()
        return weight_tensors.to(self.dtype)
