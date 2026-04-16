# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from mindie_llm.runtime.tokenizer.tokenizer_wrapper import TokenizerWrapper
from mindie_llm.modeling.model_wrapper.wrapper import BaseTokenizerWrapper
from ....utils.log.logging import logger


class AclGraphTokenizerWrapper(BaseTokenizerWrapper):
    def __init__(self, model_id: str, **kwargs):
        self.tokenizer_wrapper = TokenizerWrapper(model_id, **kwargs)
        self.config = self.tokenizer_wrapper.config
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.tokenizer = self.tokenizer_wrapper.tokenizer
        self.input_builder = self.tokenizer_wrapper.input_builder
        self.vocab_size = self.config.vocab_size
        self.tool_calls_processor = self.tokenizer_wrapper.tool_calls_processor

    def encode(self, inputs, **kwargs):
        try:
            token_ids = self.tokenizer_wrapper.encode(inputs, **kwargs)
        except Exception as e:
            logger.error(f"Error in encode: {e}")
            raise e
        return token_ids

    def decode(
        self,
        all_token_ids: list[int],
        skip_special_tokens: bool,
        use_tool_calls: bool,
        is_chat_req: bool,
        stream: bool,
        **kwargs,
    ):
        try:
            decoded_text = self.tokenizer_wrapper.decode(
                all_token_ids, skip_special_tokens, use_tool_calls, is_chat_req, stream, **kwargs
            )
        except Exception as e:
            logger.error(f"Error in decode: {e}")
            raise e
        return decoded_text
