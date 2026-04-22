# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from typing_extensions import Self

from mindie_llm.utils.log.logging import message_filter


@dataclass
class LoraConfig:
    """A base class used to store lora adapter configuration information.

    Attributes:
        r: Lora attention dimension.
        lora_alpha: Lora alpha.
        use_rslora: When set to True, which sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it
            was proven to work better. Otherwise, it will use the original default value of `lora_alpha/r`.
        rank_pattern: The mapping from layer names to ranks which are different from the default rank specified by `r`.
            For example, `{'model.decoder.layers.0.encoder_attn.k_proj': 8}`.
        alpha_pattern: The mapping from layer names to alphas which are different from
            the default alpha specified by `lora_alpha`.
            For example, `{'model.decoder.layers.0.encoder_attn.k_proj': 32}`
        target_modules: The list of module names where LoRA adapters should be inserted.
            For example, `{'q_proj', 'k_proj', 'v_proj'}`.
            Note: Some modules (e.g., `embed_tokens`, `lm_head`) are not supported with LoRA.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    lora_alpha: int = field(default=8)
    use_rslora: bool = field(default=False)
    rank_pattern: Optional[dict] = field(default_factory=dict)
    alpha_pattern: Optional[dict] = field(default_factory=dict)
    target_modules: Optional[List[str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> Self:
        """Method used to construct an object of LoraConfig class.

        This method is used to construct an object from a dictionary.

        Args:
            config_dict: A dictionary containing configuration parameters.
        """
        field_names = set(f.name for f in dataclasses.fields(cls))
        config = cls(**{k: v for k, v in config_dict.items() if k in field_names})
        config._validate()
        return config

    def _validate(self):
        """Perform security validation on the input parameters."""
        self._check_r()
        self._check_lora_alpha()
        self._check_rank_pattern()
        self._check_alpha_pattern()
        self._check_target_modules()

    def _check_r(self):
        if self.r <= 0 or self.r > 1024:
            raise ValueError("The parameter `r` must be larger than 0 and less than or equal to 1024")

    def _check_lora_alpha(self):
        if self.lora_alpha <= 0 or self.lora_alpha > 1024:
            raise ValueError("The parameter `lora_alpha` must be larger than 0 and less than or equal to 1024")

    def _check_rank_pattern(self):
        if self.rank_pattern is None:
            return
        if not isinstance(self.rank_pattern, dict):
            raise ValueError("The parameter `rank_pattern` must be a dictionary")
        for rank_key, rank_value in self.rank_pattern.items():
            if rank_value <= 0 or rank_value > 1024:
                error_msg = f"The parameter `{rank_key}` must be larger than 0 and less than or equal to 1024"
                error_msg = message_filter(error_msg)
                raise ValueError(error_msg)

    def _check_alpha_pattern(self):
        if self.alpha_pattern is None:
            return
        if not isinstance(self.alpha_pattern, dict):
            raise ValueError("The parameter `alpha_pattern` must be a dictionary")
        for alpha_key, alpha_value in self.alpha_pattern.items():
            if alpha_value <= 0 or alpha_value > 1024:
                error_msg = f"The parameter `{alpha_key}` must be larger than 0 and less than or equal to 1024"
                error_msg = message_filter(error_msg)
                raise ValueError(error_msg)

    def _check_target_modules(self):
        unsupported_modules = {"embed_tokens", "lm_head"}
        if self.target_modules:
            target_modules = set(self.target_modules)
            if target_modules & unsupported_modules:
                raise ValueError(
                    f"[INVALID LORA WEIGHT] Layers {target_modules & unsupported_modules} with LoRA "
                    + "are unsupported."
                )
