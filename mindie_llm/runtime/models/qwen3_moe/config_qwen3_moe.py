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

from mindie_llm.runtime.config.huggingface_config import HuggingFaceConfig


@dataclass
class Qwen3MoeConfig(HuggingFaceConfig):
    """Configuration class for Qwen3-Moe model.

    Extends HuggingFaceConfig with Qwen3-Moe-specific attributes.
    """

    use_qk_norm: bool = True
    is_reasoning_model: bool = True

    def __init__(self, **kwargs):
        """Initializes Qwen3-Moe configuration with optional keyword arguments."""
        super().__init__(**kwargs)
