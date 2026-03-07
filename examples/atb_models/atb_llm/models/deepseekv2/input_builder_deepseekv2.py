# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from ..base.input_builder import InputBuilder


class Deepseekv2InputBuilder(InputBuilder):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        # Thinking chain switch for weight configuration
        self.config_thinking = tokenizer.init_kwargs.get("thinking", None)

    def _apply_chat_template(self, conversation, tools_msg=None, **kwargs):
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise RuntimeError("Your transformers version is detected to be <4.34. This message indicates that this "
                               "model is not supported to run on transformers <4.34. You can upgrade transformers to "
                               "4.34 or above, or rewrite the InputBuilder provided with this model and load it in the "
                               "router.")
        if not self.tokenizer.chat_template:
            raise RuntimeError("The model does not appear to be a chat model because it is not configured with a "
                               "`chat_template`.")

        for single_conversation in conversation:
            if single_conversation.get("content", None) is None:
                single_conversation["content"] = ""

        request_enable_thinking = kwargs.get("chat_template_kwargs", {}).get("enable_thinking", None)
        if request_enable_thinking is not None:
            enable_thinking = request_enable_thinking
        else:
            enable_thinking = self.config_thinking
        if enable_thinking is not None:
            kwargs.update({"thinking": enable_thinking})

        if tools_msg:
            return self.tokenizer.apply_chat_template(conversation, tools=tools_msg.get("tools", None), **kwargs)
        return self.tokenizer.apply_chat_template(conversation, **kwargs)