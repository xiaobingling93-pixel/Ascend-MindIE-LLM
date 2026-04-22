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

from mindie_llm.runtime.models.qwen3_moe.input_builder_qwen3_moe import Qwen3MoeInputBuilder
from mindie_llm.runtime.models.base.router import BaseRouter
from mindie_llm.runtime.utils.helpers.safety.hf import safe_get_tokenizer_from_pretrained


@dataclass
class Qwen3MoeRouter(BaseRouter):
    """
    Router class for Qwen3Moe model configuration and initialization.

    This class handles the specific configuration and setup for Qwen3Moe models,
    including configuration creation, tokenizer initialization, and input builder setup.
    """

    def _get_tokenizer(self):
        """
        Creates and returns the tokenizer for Qwen3Moe model.

        Sets padding side to "left" and uses trust_remote_code flag.

        Returns:
            Tokenizer: Tokenizer for Qwen3Moe model
        """
        return safe_get_tokenizer_from_pretrained(
            self.load_config.tokenizer_path, padding_side="left", trust_remote_code=self.load_config.trust_remote_code
        )

    def _get_input_builder(self):
        """
        Creates and returns the input builder for Qwen3Moe model.

        Optionally accepts a custom chat template if provided.

        Returns:
            Qwen3MoeInputBuilder: Input builder for Qwen3Moe model
        """
        kwargs = {}
        if self.custom_chat_template:
            kwargs["chat_template"] = self.custom_chat_template
        if hasattr(self.config, "max_position_embeddings") and self.config.max_position_embeddings:
            kwargs["max_length"] = self.config.max_position_embeddings
        return Qwen3MoeInputBuilder(self.tokenizer, **kwargs)

    def _get_tool_calls_parser(self):
        """
        Returns the tool call parser identifier for Qwen3Moe.

        Returns:
            str: Identifier for the tool call parser ("qwen3")
        """
        return "qwen3"
