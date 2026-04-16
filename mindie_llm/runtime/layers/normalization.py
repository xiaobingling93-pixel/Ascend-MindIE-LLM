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

from mindie_llm.runtime.layers.custom_layer import CustomLayer
from mindie_llm.runtime.layers.parameter import BaseParameter
from mindie_llm.runtime.layers.quantization.quantization_config_base import QuantizationConfigBase
from mindie_llm.runtime.layers.quantization.quantization_method_base import QuantizationMethodBase
from mindie_llm.runtime.layers.quantization.unquantized import UnquantizedNormMethod, UnquantizedLayerNormBiasMethod
from mindie_llm.utils.log.logging import logger


class RMSNorm(CustomLayer):
    """Root Mean Square Normalization layer.

    This layer implements RMS normalization, which normalizes the input by dividing
    by the root mean square of the input. It supports quantization through the
    quantization configuration.

    Args:
        hidden_size: The size of the hidden dimension.
        eps: A small value added to the denominator for numerical stability.
            Defaults to 1e-6.
        var_hidden_size: Size of the subset of the input tensor's last dimension
            over which to compute variance. Not currently supported. Defaults to None.
        weight_dtype: The data type for the weight tensor. If None, uses the
            default dtype. Defaults to None.
        quant_config: The quantization configuration for the layer. If None,
            uses unquantized method. Defaults to None.
        prefix: Prefix string used to construct the full layer name in the state dictionary. Defaults to "".
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        var_hidden_size: int | None = None,
        weight_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfigBase | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        if var_hidden_size is not None:
            logger.warning("Passing not-null `var_hidden_size` to `RMSNorm` is not supported.")

        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.weight_dtype = weight_dtype or torch.get_default_dtype()
        self.quant_config = quant_config
        self.prefix = prefix

        if self.quant_config is None:
            self.quant_method: QuantizationMethodBase | None = UnquantizedNormMethod()
        else:
            self.quant_method = self.quant_config.get_quant_method(self, prefix=prefix)
        self.quant_method.create_weights(self, hidden_size, weight_dtype, weight_loader=self.weight_loader)

    def weight_loader(self, param: BaseParameter, loaded_weight: torch.Tensor) -> None:
        """Load weight into a parameter with parallel support.

        Args:
            param: The parameter to load the weight into.
            loaded_weight: The weight tensor read from file to be loaded into the parameter.
        """
        param.load_weight(loaded_weight)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the RMS normalization layer.

        Args:
            x: Input tensor to normalize.
            residual: Optional residual tensor to add before normalization.
                If provided, returns both normalized output and updated residual.

        Returns:
            Normalized tensor, or a tuple of (normalized_tensor, residual) if
            residual is provided.
        """
        return self.quant_method.apply(self, x, residual)

    def extra_repr(self) -> str:
        """Return a string representation of the layer configuration.

        Returns:
            A string containing the layer configuration details.
        """
        s = f"hidden_size={self.hidden_size}, eps={self.variance_epsilon}, dtype={self.weight_dtype}"
        s += f", quant_method={self.quant_method.__class__.__name__}"
        return s


class LayerNorm(CustomLayer):
    """Layer Normalization layer.

    This layer implements standard layer normalization, which normalizes the
    input across the feature dimension. Unlike RMSNorm, it always includes
    a bias term.

    Args:
        hidden_size: The size of the hidden dimension.
        eps: A small value added to the denominator for numerical stability.
            Defaults to 1e-6.
        var_hidden_size: Size of the subset of the input tensor's last dimension
            over which to compute variance. Not currently supported. Defaults to None.
        weight_dtype: The data type for the weight tensor. If None, uses the
            default dtype. Defaults to None.
        quant_config: The quantization configuration for the layer. If None,
            uses unquantized method. Defaults to None.
        prefix: Prefix string used to construct the full layer name in the state dictionary. Defaults to "".
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        var_hidden_size: int | None = None,
        weight_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfigBase | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        if var_hidden_size is not None:
            logger.warning("Passing not-null `var_hidden_size` to `RMSNorm` is not supported.")

        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.weight_dtype = weight_dtype or torch.get_default_dtype()
        self.quant_config = quant_config
        self.prefix = prefix

        if self.quant_config is None:
            self.quant_method = UnquantizedLayerNormBiasMethod()
        else:
            self.quant_method = self.quant_config.get_quant_method(self, prefix=prefix)
        self.quant_method.create_weights(self, hidden_size, weight_dtype, weight_loader=self.weight_loader)

    def weight_loader(self, param: BaseParameter, loaded_weight: torch.Tensor) -> None:
        """Load weight into a parameter with parallel support.

        Args:
            param: The parameter to load the weight into.
            loaded_weight: The weight tensor read from file to be loaded into the parameter.
        """
        param.load_weight(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the layer normalization layer.

        Args:
            x: Input tensor to normalize.

        Returns:
            Normalized tensor.
        """
        return self.quant_method.apply(self, x, self.hidden_size)
