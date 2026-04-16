# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Any, Dict, List, Optional
from ..base.input_builder import InputBuilder


class Qwen3InputBuilder(InputBuilder):
    """
    1. Supporting the function call
    2. Support for request-level closure of the think
    """

    def __init__(self, tokenizer, **kwargs):
        """Initializes the Qwen3 input builder with tokenizer and optional arguments.

        Args:
            tokenizer: The tokenizer instance used for encoding and applying chat templates.
            **kwargs: Additional keyword arguments passed to the parent InputBuilder.
        """
        super().__init__(tokenizer, **kwargs)
        # Thinking chain switch for weight configuration
        self.config_enable_thinking = tokenizer.init_kwargs.get("enable_thinking", True)

    def _apply_chat_template(
        self, conversation: List[Dict[str, str]], tools_msg: Optional[Dict[str, Any]] = None, **kwargs
    ) -> List[int]:
        """Applies the chat template with support for reasoning toggle and tool calls.

        This method extends the base implementation by:
        - Allowing request-level override of the thinking chain via `enable_thinking`.
        - Injecting tools into the tokenizer call when `tools_msg` is provided.

        Args:
            conversation: List of message dictionaries with 'role' and 'content' keys.
            tools_msg: Optional dictionary containing a 'tools' key with tool definitions.
            **kwargs: Additional arguments passed to the tokenizer's apply_chat_template,
                including 'chat_template_kwargs' which may contain 'enable_thinking'.

        Returns:
            List[int]: Tokenized input IDs after applying the chat template.

        Raises:
            RuntimeError: If the tokenizer lacks `apply_chat_template` or `chat_template`.
        """
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise RuntimeError(
                "Your transformers version is detected to be <4.34. This message indicates that this "
                "model is not supported to run on transformers <4.34. You can upgrade transformers to "
                "4.34 or above, or rewrite the InputBuilder provided with this model and load it in the "
                "router."
            )
        if not self.tokenizer.chat_template:
            raise RuntimeError(
                "The model does not appear to be a chat model because it is not configured with a `chat_template`."
            )
        # Request-level switch > Weight tokenizer_config
        request_enable_thinking = kwargs.get("chat_template_kwargs", {}).get("enable_thinking", None)
        if request_enable_thinking is not None:
            enable_thinking = request_enable_thinking
        else:
            enable_thinking = self.config_enable_thinking
        kwargs["enable_thinking"] = enable_thinking
        if tools_msg:
            return self.tokenizer.apply_chat_template(conversation, tools=tools_msg.get("tools", None), **kwargs)
        return self.tokenizer.apply_chat_template(conversation, **kwargs)
