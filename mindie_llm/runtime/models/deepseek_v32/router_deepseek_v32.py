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

from mindie_llm.runtime.models.deepseek_v3.router_deepseek_v3 import DeepseekV3Router
from mindie_llm.runtime.models.deepseek_v3.deepseek_v3 import DeepseekV3ForCausalLM
from mindie_llm.runtime.models.deepseek_v3.deepseek_v3_mtp import DeepseekV3MTP
from mindie_llm.runtime.models.deepseek_v3.config_deepseek_v3 import DeepseekV3Config
from mindie_llm.runtime.models.deepseek_v32.input_builder_deepseekv32 import Deepseekv32InputBuilder


@dataclass
class DeepseekV32Router(DeepseekV3Router):
    """
    Router class for DeepSeek V3.2 model configuration and initialization.

    This class extends the DeepseekV3Router to handle specific configuration and
    initialization for the DeepSeek V3.2 model variant. It adjusts the model type
    to match the underlying DeepSeek V3 implementation while maintaining
    V3.2-specific functionality.
    """

    def _get_input_builder(self):
        """
        Creates and returns the input builder for DeepSeek V3.2 model.

        Returns:
            Deepseekv32InputBuilder: Input builder specific to DeepSeek V3.2 model
        """
        kwargs = {}
        if self.custom_chat_template:
            kwargs["chat_template"] = self.custom_chat_template
        if hasattr(self.config, "max_position_embeddings") and self.config.max_position_embeddings:
            kwargs["max_length"] = self.config.max_position_embeddings
        return Deepseekv32InputBuilder(self.tokenizer, **kwargs)

    def _get_model_cls(self):
        """Returns model cls of DeepSeek V3.2, where we reuse DeepSeek V3 model."""
        return DeepseekV3ForCausalLM

    def _get_draft_cls(self):
        """Returns mtp model cls of DeepSeek V3.2, where we reuse DeepSeek V3 mtp model."""
        return DeepseekV3MTP

    def _get_config_cls(self):
        """Returns config cls of DeepSeek V3.2, where we reuse DeepSeek V3 config."""
        return DeepseekV3Config

    def _get_tool_calls_parser(self):
        """
        Returns the tool call parser identifier for DeepSeek V3.2.

        Returns:
            str: Identifier "deepseek_v32" for tool call parsing
        """
        return "deepseek_v32"
