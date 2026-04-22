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
from mindie_llm.runtime.models.deepseek_v3.input_builder_deepseek_v3 import DeepseekV3InputBuilder


@dataclass
class DeepseekV3Router(BaseRouter):
    """
    Router class for DeepSeek V3 model configuration and initialization.

    This class handles the specific configuration and setup for DeepSeek V3 models,
    including model type conversion, configuration validation, and component initialization.
    """

    def _get_tokenizer(self):
        """
        Creates and returns the tokenizer for DeepSeek V3 model.

        Sets the pad token to be the same as the EOS token.

        Returns:
            Tokenizer: Tokenizer for DeepSeek V3 model
        """
        tokenizer = safe_get_tokenizer_from_pretrained(
            self.load_config.tokenizer_path,
            padding_side="left",
            trust_remote_code=False,
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def _get_input_builder(self):
        """
        Creates and returns the input builder for DeepSeek V3 model.

        Returns:
            DeepseekV3InputBuilder: Input builder for DeepSeek V3 model
        """
        kwargs = {}
        if self.custom_chat_template:
            kwargs["chat_template"] = self.custom_chat_template
        if hasattr(self.config, "max_position_embeddings") and self.config.max_position_embeddings:
            kwargs["max_length"] = self.config.max_position_embeddings
        return DeepseekV3InputBuilder(self.tokenizer, **kwargs)

    def _get_tool_calls_parser(self):
        """
        Returns the tool call parser identifier for DeepSeek V3.

        Returns:
            str: Identifier for the tool call parser ("deepseek_v3")
        """
        return "deepseek_v3"
