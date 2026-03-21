# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from dataclasses import dataclass

import torch

from mindie_llm.runtime.model_runner.input_buffer import input_buffer
from mindie_llm.runtime.model_runner.forward_metadata.module_metadata import ModuleMetadata


@dataclass
class MtpMetadata(ModuleMetadata):
    last_hidden_states: torch.Tensor = None
    draft_token_indices = None

    @staticmethod
    def from_model_input(model_inputs):
        return MtpMetadata(
            last_hidden_states=model_inputs.last_hidden_states
        )

    @staticmethod
    def register_buffer(max_num_token, device, hf_config):
        input_buffer.register("last_hidden_states",
            torch.zeros((max_num_token, hf_config.hidden_size),
            dtype=hf_config.torch_dtype, device=device))
        
    def copy(self, num_actual_tokens, num_tokens):
        if self.last_hidden_states is not None:
            input_buffer.get("last_hidden_states")[:num_actual_tokens, :].copy_(
                self.last_hidden_states[:num_actual_tokens, :]
            )
            self.last_hidden_states = input_buffer.get("last_hidden_states")[:num_tokens, :]