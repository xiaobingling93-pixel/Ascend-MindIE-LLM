# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from dataclasses import dataclass

from mindie_llm.runtime.config.mindie_llm_config import MindIELLMConfig
from mindie_llm.runtime.utils.helpers.parameter_validators import BooleanParameterValidator


@dataclass
class ModelDescriptor:
    """Represents model configuration descriptor with validation capabilities.

    Attributes:
        is_flashcomm_supported (bool): Indicates whether FlashComm acceleration is supported for this model.
    """

    is_flashcomm_supported: bool = False

    def __post_init__(self):
        """Performs validation after dataclass initialization."""
        self.validate()

    @classmethod
    def from_config(cls, mindie_llm_config: MindIELLMConfig) -> "ModelDescriptor":
        """Creates ModelDescriptor instance from configuration dictionary."""
        model_descriptor = cls(is_flashcomm_supported=False)
        return model_descriptor

    def validate(self) -> None:
        """Validates all attributes against their respective validators."""
        validators = {
            "is_flashcomm_supported": BooleanParameterValidator(),
        }

        for attr, validator in validators.items():
            value = getattr(self, attr)
            validator.validate(value, attr)
