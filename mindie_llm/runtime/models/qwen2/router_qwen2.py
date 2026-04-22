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

from mindie_llm.runtime.models.base.router import BaseRouter
from mindie_llm.runtime.utils.helpers.safety.hf import safe_get_tokenizer_from_pretrained


@dataclass
class Qwen2Router(BaseRouter):
    """
    Router class specifically designed for Qwen2 models.

    This class extends BaseRouter with Qwen2-specific functionality, including:
    - Tokenizer configuration specific to Qwen2
    - Specialized input builder (Qwen2InputBuilder)
    - Tool call parser identification
    """

    def _get_tokenizer(self):
        """
        Get the tokenizer for Qwen2 models.

        This method configures the tokenizer with Qwen2-specific settings:
        - Left padding (for batch processing)
        - Trust remote code if specified (security consideration)

        Returns:
            Tokenizer: A configured tokenizer for Qwen2 models
        """
        return safe_get_tokenizer_from_pretrained(
            self.load_config.tokenizer_path, padding_side="left", trust_remote_code=self.load_config.trust_remote_code
        )

    # NOTE InputBuilder for qwen2 will be implemented later

    def _get_tool_calls_parser(self):
        raise NotImplementedError("Subclass must implement _get_tool_calls_parser")
