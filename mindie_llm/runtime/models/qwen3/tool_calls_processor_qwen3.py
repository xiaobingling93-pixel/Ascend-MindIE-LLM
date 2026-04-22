# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import re
from typing import Pattern

from ..base.tool_calls_processor import ToolCallsProcessorWithXml, ToolCallsProcessorManager


@ToolCallsProcessorManager.register_module(["qwen3", "qwen3_moe", "hermes"])
class ToolCallsProcessorQwen3(ToolCallsProcessorWithXml):
    """Tool call processor for Qwen3, Qwen3-MoE, and Hermes models.

    This processor handles tool calls enclosed between '<tool_call>' start and end tokens,
    which are specific to the Qwen3 model family.
    """

    def __init__(self, tokenizer=None):
        """Initializes the Qwen3 tool call processor.

        Args:
            tokenizer: Optional tokenizer instance used for decoding token IDs.
        """
        super().__init__(tokenizer)
        self._tool_calls_regex = re.compile(r"<tool_call>\s*({.*?})\s*</tool_call>", re.DOTALL)

    @property
    def tool_call_start_token(self) -> str:
        """Returns the start token string for tool calls in Qwen3."""
        return "<tool_call>"  # start_token of qwen3

    @property
    def tool_call_end_token(self) -> str:
        """Returns the end token string for tool calls in Qwen3."""
        return "</tool_call>"  # end_token of qwen3

    @property
    def tool_call_start_token_id(self) -> int:
        """Returns the token ID for the tool call start token in Qwen3."""
        return 151657  # start_token_id of qwen3

    @property
    def tool_call_end_token_id(self) -> int:
        """Returns the token ID for the tool call end token in Qwen3."""
        return 151658  # end_token_id of qwen3

    @property
    def tool_call_regex(self) -> Pattern:
        """Returns the compiled regex pattern for extracting tool call JSON content."""
        return self._tool_calls_regex
