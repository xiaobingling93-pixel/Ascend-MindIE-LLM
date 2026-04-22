# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import json
from ..qwen3.input_builder_qwen3 import Qwen3InputBuilder


class Qwen3MoeInputBuilder(Qwen3InputBuilder):
    """
    1. Inherits from Qwen3InputBuilder
    2. Support JSON loading for tool call
    """

    def _apply_chat_template(self, conversation, tools_msg=None, **kwargs):
        """Processes tool call arguments in conversation assistant messages, then applies the base chat template.

        This method extends the parent class implementation by:
        - Iterating through assistant role messages to standardize tool call arguments format.
        - Safely parsing JSON string-formatted tool call arguments into native Python dictionaries.
        - Gracefully skipping malformed or non-JSON tool call arguments without breaking execution.
        - Retaining all original functionality of the parent class `_apply_chat_template` method.

        Args:
            conversation: List of message dictionaries with 'role' and 'content' keys.
            tools_msg: Optional dictionary containing a 'tools' key with tool definitions.
            **kwargs: Additional arguments passed to the tokenizer's apply_chat_template,
                including 'chat_template_kwargs' which may contain 'enable_thinking'.

        Returns:
            List[int]: Tokenized input IDs after applying the chat template.
        """
        for message in conversation:
            if message.get("role", "user") not in ["assistant"]:
                continue

            tool_calls = message.get("tool_calls", [])
            if not isinstance(tool_calls, list):
                continue

            for tool_call in tool_calls:
                try:
                    function = tool_call.get("function", {})
                    function["arguments"] = json.loads(function.get("arguments", ""))

                except (TypeError, ValueError):
                    continue

        return super()._apply_chat_template(conversation, tools_msg, **kwargs)
