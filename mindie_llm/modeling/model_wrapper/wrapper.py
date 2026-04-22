# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from abc import abstractmethod
from typing import Iterable

import numpy as np


class ModelWrapper:
    @abstractmethod
    def forward(self, model_inputs, npu_cache=None, **kwargs):
        pass

    @abstractmethod
    def generate_position_ids(self, input_ids: np.ndarray) -> Iterable:
        pass


class BaseTokenizerWrapper:
    def tokenize(self, inputs, **kwargs):
        pass

    def detokenize(
        self,
        all_token_ids: list[int],
        skip_special_tokens: bool,
        use_tool_calls: bool,
        is_chat_req: bool,
        stream: bool,
        **kwargs,
    ):
        """
        detokenize
        Args:
            all_token_ids: all_token_ids.
            skip_special_tokens: skip_special_tokens.
            use_tool_calls: use_tool_calls.
            is_chat_req: is_chat_req.
            stream: stream.
            kwargs: prev_decode_index、curr_decode_index、metadata, metadata has req_enable_thinking、
            current_tool_name_sent、current_tool_arguments_sent、current_tool_id.

        Returns:
            dict: detokenize result, may contain content\rreasoning_content\tool_calls or {}

        Example:
            result {"content": "你好","reasoning_content": "\n好的，用户发..."}
        """
        pass
