# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Callable, Any

import torch
from torch.nn import Parameter

from mindie_llm.utils.log.logging import logger


class BaseParameter(Parameter):
    """Base parameter class for model weights.

    This class extends PyTorch's Parameter class to provide additional
    functionality for weight loading and attribute management. It disables
    gradient computation by default.

    Args:
        data: The tensor data for the parameter.
    """

    def __new__(cls, data: torch.Tensor | None, **kwargs):
        return super().__new__(cls, data=data, requires_grad=False)

    def __init__(self, data: torch.Tensor) -> None:
        """Initialize the BaseParameter.

        Args:
            data: The tensor data for the parameter.
        """
        self._weight_loader = None
        self._extra_attr_keys = []

    @property
    def weight_loader(self) -> Callable:
        """Return the weight loader callable function."""
        return self._weight_loader

    @weight_loader.setter
    def weight_loader(self, value: Callable) -> None:
        """Set the weight loader function.

        Args:
            value: The callable function to use for loading weights.
        """
        self._weight_loader = value

    @staticmethod
    def _check_and_copy(param_data: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        """Check shape compatibility and copy loaded weight to parameter data.
        Raise ValueError if the shapes of `param_data` and `loaded_weight` don't match.
        """
        if param_data.shape != loaded_weight.size():
            raise ValueError(
                f"Tried to load weights of size {loaded_weight.size()} to a parameter of size {param_data.shape}"
            )
        param_data.copy_(loaded_weight)

    def add_attrs(self, attrs: dict[str, Any]) -> None:
        """Add extra attributes to the parameter.

        Args:
            attrs: Dictionary of attribute names and values to add.

        Raises:
            KeyError: If trying to overwrite an existing attribute.
        """
        for key, value in attrs.items():
            if key in self._extra_attr_keys:
                raise KeyError(f"Overwriting existing attribute {key} for Paramter.")
            setattr(self, key, value)
            self._extra_attr_keys.append(key)

    def load_weight(self, loaded_weight: torch.Tensor) -> None:
        """Load weight tensor read from file into the parameter. It will check shape compatibility.

        Args:
            loaded_weight: The weight tensor read from file to be loaded into the parameter.
        """
        self._check_and_copy(self.data, loaded_weight)

    def check_required_attr(self, attrs: list[str]) -> None:
        """Check if all required attributes are present.

        Args:
            attrs: List of attribute names to check.

        Raises:
            AttributeError: If any required attribute is missing.
        """
        for attr in attrs:
            if attr not in self._extra_attr_keys:
                raise AttributeError(f"`{attr}` is not defined in {self.__class__.__name__}")


class RowParameter(BaseParameter):
    """Parameter class for row-parallel weight loading.

    This class extends BaseParameter to support loading weights that are
    partitioned along the input dimension for tensor parallelism.
    The `input_dim` attribute must be defined on the parameter instance.
    """

    def load_row_parallel_weight(
        self,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        loaded_weight_shard_offset: int | None = None,
        loaded_weight_shard_size: int | None = None,
    ) -> None:
        """Load row-parallel weight for tensor parallel sharding.

        Supports uneven partitioning when num_heads (or input dimension) is not
        divisible by tp_size. When param shard is larger than loaded shard,
        the remainder is zero-padded.

        Args:
            loaded_weight: The full weight tensor read from file to load from.
            tp_rank: The tensor parallel rank of the current process.
            loaded_weight_shard_offset: Offset in loaded_weight along input_dim for this rank.
            loaded_weight_shard_size: Size of the slice to load from loaded_weight.
        """
        self.check_required_attr(["input_dim"])

        shard_size = self.data.shape[self.input_dim]
        if loaded_weight_shard_offset is None:
            loaded_weight_shard_offset = tp_rank * shard_size
        if loaded_weight_shard_size is None:
            loaded_weight_shard_size = shard_size
        # Copy the overlapping region between param and loaded weight
        param_data_with_value = self.data.narrow(self.input_dim, 0, min(shard_size, loaded_weight_shard_size))
        loaded_weight = loaded_weight.narrow(self.input_dim, loaded_weight_shard_offset, loaded_weight_shard_size)
        self._check_and_copy(param_data_with_value, loaded_weight)

        if shard_size > loaded_weight_shard_size:
            # Zero-pad remainder when param partition is larger (uneven split)
            padding_size = list(self.data.shape)
            padding_size[self.input_dim] = shard_size - loaded_weight_shard_size
            param_data_with_padding = self.data.narrow(
                self.input_dim, loaded_weight_shard_size, padding_size[self.input_dim]
            )
            padding_tensor = torch.zeros(padding_size, dtype=loaded_weight.dtype)
            self._check_and_copy(param_data_with_padding, padding_tensor)
        elif shard_size < loaded_weight_shard_size:
            logger.warning(
                f"The `shard_size` of loaded weight ({loaded_weight_shard_size}) is larger than"
                f" the `shard_size` of the target parameter ({shard_size})."
            )

    def load_expert_row_parallel_weight(self, loaded_weight: torch.Tensor, expert_id: int, tp_rank: int) -> None:
        """Load row-parallel weight for a specific expert in MoE models.

        Args:
            loaded_weight: The full weight tensor read from file to load from.
            expert_id: The ID of the expert to load weights for.
            tp_rank: The tensor parallel rank of the current process.
        """
        self.check_required_attr(["input_dim"])

        expert_data = self.data[expert_id]
        shard_size = expert_data.shape[self.input_dim]
        loaded_weight = loaded_weight.narrow(self.input_dim, tp_rank * shard_size, shard_size)

        self._check_and_copy(expert_data, loaded_weight)


class ColumnParameter(BaseParameter):
    """Parameter class for column-parallel weight loading.

    This class extends BaseParameter to support loading weights that are
    partitioned along the output dimension for tensor parallelism.
    The `output_dim` attribute must be defined on the parameter instance.
    """

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor, tp_rank: int) -> None:
        """Load column-parallel weight for tensor parallel sharding.

        Args:
            loaded_weight: The full weight tensor read from file to load from.
            tp_rank: The tensor parallel rank of the current process.
        """
        self.check_required_attr(["output_dim"])

        shard_size = self.data.shape[self.output_dim]
        loaded_weight = loaded_weight.narrow(self.output_dim, tp_rank * shard_size, shard_size)

        self.load_weight(loaded_weight)

    def load_merged_column_weight(
        self, loaded_weight: torch.Tensor, tp_rank: int, shard_offset: int, shard_size: int
    ) -> None:
        """Load merged column weight with specific shard offset and size.

        Args:
            loaded_weight: The full weight tensor read from file to load from.
            tp_rank: The tensor parallel rank of the current process.
            shard_offset: The offset for the shard in the output dimension.
            shard_size: The size of the shard in the output dimension.
        """
        self.check_required_attr(["output_dim"])

        param_data = self.data
        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.narrow(self.output_dim, tp_rank * shard_size, shard_size)

        self._check_and_copy(param_data, loaded_weight)

    def load_expert_column_parallel_weight(
        self, loaded_weight: torch.Tensor, expert_id: int, tp_rank: int, shard_offset: int, shard_size: int
    ) -> None:
        """Load column weight for a specific expert in MoE models.

        Args:
            loaded_weight: The full weight tensor read from file to load from.
            expert_id: The ID of the expert to load weights for.
            tp_rank: The tensor parallel rank of the current process.
            shard_offset: The offset for the shard in the output dimension.
            shard_size: The size of the shard in the output dimension.
        """
        self.check_required_attr(["output_dim"])

        expert_data = self.data[expert_id]
        expert_data = expert_data.narrow(self.output_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.narrow(self.output_dim, tp_rank * shard_size, shard_size)

        self._check_and_copy(expert_data, loaded_weight)

    def load_qkv_weight(
        self,
        loaded_weight: torch.Tensor,
        shard_offset: int,
        shard_size: int,
        loaded_weight_shard_offset: int | None,
        loaded_weight_shard_size: int | None,
    ) -> None:
        """Load QKV weight with special handling for key-value head replication.

        Supports total_num_heads not divisible by tp_size. When param shard is larger
        than loaded shard, the remainder is zero-padded.

        Args:
            loaded_weight: The full weight tensor read from file to load from.
            shard_offset: The offset for the shard in the output dimension (param).
            shard_size: The size of the shard in the output dimension (param).
            loaded_weight_shard_offset: Offset in loaded_weight for this rank's slice.
            loaded_weight_shard_size: Size of the slice to load from loaded_weight.
        """
        self.check_required_attr(["output_dim"])

        param_data = self.data
        # Copy the overlapping region
        param_data_with_value = param_data.narrow(
            self.output_dim, shard_offset, min(shard_size, loaded_weight_shard_size)
        )
        loaded_weight = loaded_weight.narrow(self.output_dim, loaded_weight_shard_offset, loaded_weight_shard_size)
        self._check_and_copy(param_data_with_value, loaded_weight)

        if shard_size > loaded_weight_shard_size:
            # Zero-pad remainder when param size is larger than the partition of `loaded_weight`
            padding_size = list(param_data.shape)
            padding_size[self.output_dim] = shard_size - loaded_weight_shard_size
            param_data_with_padding = param_data.narrow(
                self.output_dim, shard_offset + loaded_weight_shard_size, padding_size[self.output_dim]
            )
            padding_tensor = torch.zeros(padding_size, dtype=loaded_weight.dtype)
            self._check_and_copy(param_data_with_padding, padding_tensor)
        elif shard_size < loaded_weight_shard_size:
            logger.warning(
                f"The `shard_size` of loaded weight ({loaded_weight_shard_size}) is larger than"
                f" the `shard_size` of the target parameter ({shard_size})."
            )


class ModelWeightParameter(ColumnParameter, RowParameter):
    """Parameter class that supports both row and column parallel weight loading.

    This class inherits from both RowParameter and ColumnParameter, allowing
    it to handle weights that may be partitioned along either dimension.
    """


class BiasParameter(ColumnParameter, RowParameter):
    """Parameter class for bias terms with special row-parallel loading behavior.

    For row-parallel loading, only rank 0 loads the bias; other ranks zero it out.
    """

    def load_row_parallel_weight(
        self,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        loaded_weight_shard_offset: int | None = None,
        loaded_weight_shard_size: int | None = None,
    ) -> None:
        """Load row-parallel bias weight (only rank 0 loads, others zero out).

        Args:
            loaded_weight: The full weight tensor read from file to load from.
            tp_rank: The tensor parallel rank of the current process.
            loaded_weight_shard_offset: Offset in loaded_weight for this rank.
            loaded_weight_shard_size: Size of slice to load.
        """
        if tp_rank == 0:
            super().load_row_parallel_weight(
                loaded_weight, tp_rank, loaded_weight_shard_offset, loaded_weight_shard_size
            )
        else:
            self.data.zero_()


class ScalerParameter(BaseParameter):
    """Parameter class for scale factors used in quantization."""

    pass


class PerTensorScaleParameter(ColumnParameter):
    """Parameter class for per-tensor scale factors used in quantization.

    This specialized parameter type handles dequantization scales, requiring
    additional processing when weights are generated in float16 but used in
    bfloat16 precision.
    """

    pass


class ExpertsParameter(BaseParameter):
    """Parameter class for experts' parameters in MoE models."""

    def load_expert_weight(
        self,
        loaded_weight: torch.Tensor,
        expert_id: int,
    ) -> None:
        """Load weight for a specific expert in MoE models.

        Args:
            loaded_weight: The full weight tensor read from file to load from.
            expert_id: The ID of the expert to load weights for.
        """

        expert_data = self.data[expert_id]
        self._check_and_copy(expert_data, loaded_weight)
