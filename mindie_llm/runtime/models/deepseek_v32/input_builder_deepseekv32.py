# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from mindie_llm.runtime.models.base.input_builder import InputBuilder
from mindie_llm.runtime.models.deepseek_v32.encoding_deepseek_v32 import encode_messages

TOOLS = "tools"


class Deepseekv32InputBuilder(InputBuilder):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        # Thinking chain switch for weight configuration
        self.config_thinking = tokenizer.init_kwargs.get("thinking", None)

    def _apply_chat_template(self, conversation, tools_msg=None, **kwargs):
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise RuntimeError(
                "Your transformers version is detected to be <4.34. This message indicates that this "
                "model is not supported to run on transformers <4.34. You can upgrade transformers to "
                "4.34 or above, or rewrite the InputBuilder provided with this model and load it in the "
                "router."
            )

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

        thinking_mode = "thinking"
        if not enable_thinking:
            thinking_mode = "chat"

        messages = conversation.copy()
        drop_thinking = True
        if tools_msg is not None and tools_msg[TOOLS]:
            messages.insert(0, {"role": "system"})
            messages[0][TOOLS] = tools_msg[TOOLS]
            drop_thinking = False
        encode_config = dict(thinking_mode=thinking_mode, drop_thinking=drop_thinking)
        prompt_str = encode_messages(messages, **encode_config)
        prompt_ids = self.tokenizer.encode(
            prompt_str, add_special_tokens=False, truncation=True, max_length=self.max_length
        )
        return prompt_ids
