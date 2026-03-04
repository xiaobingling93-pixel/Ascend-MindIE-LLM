# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from typing import Optional

import numpy as np

from .generation_metadata import GenerationParams

MAX_GEN_LEN = 512

SAMPLING_DTYPE = np.dtype([
    ('repetition_penalty', np.float64),
    ('frequency_penalty', np.float64),
    ('presence_penalty', np.float64),
    ('temperature', np.float64),
    ('top_k', np.float64),
    ('top_p', np.float64),
    ('do_sample', np.float64),
    ('top_logprobs', np.float64)
])


class Request:
    def __init__(self,
                 req_id: int,
                 seq_id: int,
                 input_ids: np.ndarray,
                 generation_params: GenerationParams,
                 has_sampling: bool = False,
                 sampling_params: Optional[np.ndarray] = None,
                 split_start_position: int = 0,
                 split_end_position: int = 0,
                 last_prompt: bool = True):
        self.req_id = req_id
        self.reserved_seq_ids = []
        self.input_ids = input_ids

        self.adapter_id = generation_params.adapter_id
        self.best_of = generation_params.best_of
        self.has_logprobs = generation_params.logprobs
        self.ignore_eos = generation_params.ignore_eos
        self.include_stop_str_in_output = generation_params.include_stop_str_in_output
        self.length_penalty = generation_params.length_penalty
        self.max_new_tokens = generation_params.max_new_tokens
        self.n = generation_params.n
        self.seed = generation_params.seed
        self.skip_special_tokens = generation_params.skip_special_tokens
        self.stop_strings = generation_params.stop_strings
        self.stop_token_ids = generation_params.stop_token_ids
        self.use_beam_search = generation_params.use_beam_search

        self.block_tables = None
        self.has_sampling = has_sampling
        self.sampling_params = sampling_params
        self.input_length = np.size(self.input_ids)
        self.dp_rank_id = 0
        self.num_sequence_blocks = 0
        self.sp_tokens = None
        self.sp_rank_id = None
        
        # mix 
        self.split_start_position = split_start_position
        self.split_end_position = split_end_position
        self.last_prompt = last_prompt

        self.sequences = {seq_id: Sequence(seq_id)}
        self.completed = []

    @classmethod
    def from_warmup(cls, input_len: int, max_output_len: int, warmup_topk_size: int = 1000,
                    enable_warmup_sampling: bool = False):
        req_id = 0
        input_ids = np.ones(input_len, dtype=np.int64)
        sampling_params_ins = None
        if enable_warmup_sampling:
            sampling_params_ins = np.array([(1.1, 0.1, 0.1, 0.9, warmup_topk_size, 0.8, True, False)], SAMPLING_DTYPE)
            return cls(req_id, req_id, input_ids, GenerationParams(max_new_tokens=max_output_len),
                    split_start_position=0, split_end_position=0,
                    has_sampling=True, sampling_params=sampling_params_ins)
        else:
            return cls(req_id, req_id, input_ids, GenerationParams(max_new_tokens=max_output_len),
                    split_start_position=0, split_end_position=0)

    @classmethod
    def request_from_token(cls, input_ids, sampling_params, generation_params=None, req_id=0, seq_id=None):
        if seq_id is None:
            seq_id = req_id
        input_ids = np.array(input_ids, dtype=np.int64)
        has_sampling = True if sampling_params is not None else False
        return cls(req_id, seq_id, input_ids, generation_params, has_sampling, sampling_params)


class Sequence:
    def __init__(self, seq_id):
        self.seq_id = seq_id

        self.block_tables = None
        self.kv_length = 0

        self.cumulative_logprobs = 0
        self.eos_flag = 0
        self.logprobs = []
        self.out_token_list = []
        self.top_logprobs = []
        self.truncation_indices = []