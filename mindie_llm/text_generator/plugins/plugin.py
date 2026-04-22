# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from ..utils.sampling_output import SamplingOutput


class Plugin(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.pad_token_id = -1

    def reshape_speculative_outputs(
        self,
        sampling_output: SamplingOutput,
        next_tokens_indices: List[List[int]],
        num_new_tokens: Optional[List[int]] = None,
    ) -> None:
        if num_new_tokens is None:
            num_new_tokens = [len(indices) for indices in next_tokens_indices]
        max_num_tokens = max(num_new_tokens)
        sequence_ids = []
        parent_sequence_ids = []
        group_indices = []
        token_ids = []
        logprobs = []
        top_token_ids = []
        top_logprobs = []
        for i, indices in enumerate(next_tokens_indices):
            if sampling_output.sequence_ids is not None:
                sequence_ids.append(sampling_output.sequence_ids[indices[0]])
                parent_sequence_ids.append(sampling_output.parent_sequence_ids[indices[0]])
            group_indices.append((i, i + 1))
            padded_array = np.pad(
                sampling_output.token_ids[indices],
                pad_width=(0, max_num_tokens - len(indices)),
                constant_values=self.pad_token_id,
            )
            token_ids.append(padded_array)
            padded_array = np.pad(
                sampling_output.logprobs[indices], pad_width=(0, max_num_tokens - len(indices)), constant_values=-9999.0
            )
            logprobs.append(padded_array)
            if 0 not in sampling_output.top_token_ids.shape:
                padded_array = np.pad(
                    sampling_output.top_token_ids[indices],
                    pad_width=((0, max_num_tokens - len(indices)), (0, 0)),
                    constant_values=self.pad_token_id,
                )
                top_token_ids.append(padded_array)
                padded_array = np.pad(
                    sampling_output.top_logprobs[indices],
                    pad_width=((0, max_num_tokens - len(indices)), (0, 0)),
                    constant_values=-9999.0,
                )
                top_logprobs.append(padded_array)
            else:
                top_token_ids.append(np.zeros((max_num_tokens, 0), dtype=np.int64))
                top_logprobs.append(np.zeros((max_num_tokens, 0), dtype=np.float32))
        sampling_output.sequence_ids = np.array(sequence_ids)
        sampling_output.parent_sequence_ids = np.array(parent_sequence_ids)
        sampling_output.group_indices = group_indices
        sampling_output.token_ids = np.array(token_ids, dtype=np.int64)
        sampling_output.logprobs = np.array(logprobs)
        sampling_output.top_token_ids = np.array(top_token_ids)
        sampling_output.top_logprobs = np.array(top_logprobs)
        sampling_output.num_new_tokens = np.array(num_new_tokens)

    @abstractmethod
    def model_inputs_update(self, model_inputs, input_metadata, sampling_metadata, cache_ids, input_len_mask, **kwargs):
        pass

    @abstractmethod
    def sample_preprocess(self, logits, result, sampling_metadata, input_metadata):
        pass

    @abstractmethod
    def plugin_verify(self, sampling_output: SamplingOutput, cache_ids: np.ndarray, result):
        pass

    @abstractmethod
    def plugin_cache_update(self, cache_ids, sampling_output, la_cache_input, is_prefill=False):
        pass

    @abstractmethod
    def plugin_cache_clear(self, cache_ids, finish_reason):
        pass
