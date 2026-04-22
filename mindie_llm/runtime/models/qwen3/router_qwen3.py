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

from mindie_llm.runtime.models.qwen3.input_builder_qwen3 import Qwen3InputBuilder
from mindie_llm.runtime.models.qwen2.router_qwen2 import Qwen2Router


@dataclass
class Qwen3Router(Qwen2Router):
    """
    Router class specifically designed for Qwen3 models.

    This class extends BaseRouter with Qwen3-specific functionality, including:
    - Tokenizer configuration specific to Qwen3
    - Specialized input builder (Qwen3InputBuilder)
    - Tool call parser identification
    """

    def _get_input_builder(self):
        """
        Get the input builder for Qwen3 models.

        This method:
        1. Creates kwargs for the input builder
        2. Adds custom chat_template and max_position_embeddings if available
        3. Returns a Qwen3InputBuilder instance

        Returns:
            Qwen3InputBuilder: An input builder configured for Qwen3 models
        """
        kwargs = {}
        if self.custom_chat_template:
            kwargs["chat_template"] = self.custom_chat_template
        if hasattr(self.config, "max_position_embeddings") and self.config.max_position_embeddings:
            kwargs["max_length"] = self.config.max_position_embeddings
        return Qwen3InputBuilder(self.tokenizer, **kwargs)

    def _get_tool_calls_parser(self):
        """
        Get the tool call parser identifier for Qwen3 models.

        Returns:
            str: The identifier for Qwen3's tool call parser ("qwen3")
        """
        return "qwen3"
