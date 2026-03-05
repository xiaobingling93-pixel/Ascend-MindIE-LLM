# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from mindie_llm.text_generator.utils.input_metadata import InputMetadata
from ...utils.log.logging import logger


@dataclass
class GenerationOutput:
    """
    Args:
        trace_ids:  A 1-d array of the request ids sent by the users. It can also be the request ids set by
            scheduler when users do not set request ids.
        sequence_ids： A list of 1-d array of integers.
        parent_sequence_ids: A list of 1-d array of the parent sequence id of `sequence_ids`.
        token_ids: An array of chosen token ids, whose shape is (num_seqs, num_parallel_tokens).
        logprobs: An array of log-probabilities of the `token_ids`.
        top_token_ids: An array of top token ids whose number is specified by the input `top_logprobs` parameter. Its
            shape is (num_seqs, num_parallel_tokens, num_top_tokens).
        top_logprobs: An array of top log-probabilities of the `top_token_ids`.
        num_top_tokens: A 1-d array of the original number of `top_logprobs` used to truncate the `top_token_ids` and
            `top_logprobs`.
    """
    sequence_ids: np.ndarray
    parent_sequence_ids: np.ndarray
    group_indices: List[Tuple[int, int]]
    token_ids: Optional[Union[List[np.ndarray], np.ndarray]] = None
    logprobs: Optional[Union[List[np.ndarray], np.ndarray]] = None
    top_token_ids: Optional[Union[List[List[np.ndarray]], np.ndarray]] = None
    top_logprobs: Optional[Union[List[List[np.ndarray]], np.ndarray]] = None
    num_new_tokens: Optional[np.ndarray] = None
    num_top_tokens: Optional[np.ndarray] = None
    cumulative_logprobs: Optional[np.ndarray] = None
    finish_reason: Optional[np.ndarray] = None
    truncation_indices: Optional[np.ndarray] = None
    current_token_indices: Optional[List[int]] = None
    eos_info: Optional[np.ndarray] = None
    trace_ids: Optional[List[Any]] = None
    simulator_ids: Optional[List[Any]] = None

    @classmethod
    def make_empty(cls) -> 'GenerationOutput':
        empty_output = cls(
            sequence_ids=np.zeros((0,), dtype=np.int64),
            parent_sequence_ids=np.zeros((0,), dtype=np.int64),
            group_indices=[],
            token_ids=np.zeros((0, 0), dtype=np.int64),
            logprobs=np.zeros((0, 0), dtype=np.float32),
            top_token_ids=np.zeros((0, 0, 0), dtype=np.int64),
            top_logprobs=np.zeros((0, 0, 0), dtype=np.float32),
            num_new_tokens=np.zeros((0,), dtype=np.int64),
            num_top_tokens=np.zeros((0,), dtype=np.int64),
            cumulative_logprobs=np.zeros((0,), dtype=np.float32),
            finish_reason=np.zeros((0,), dtype=np.int32),
            truncation_indices=np.zeros((0,), dtype=np.int_),
            eos_info=np.zeros((0,), dtype=np.int64)
        )
        return empty_output

    def collate(self) -> None:
        self.eos_info = np.copy(
            np.array([self.finish_reason, self.num_new_tokens], dtype=np.int64).T, order='C')

        self.token_ids = self.token_ids.astype(np.int64)

        if self.logprobs is None:
            self.logprobs = np.zeros(self.token_ids.shape, dtype=np.float32)
        else:
            self.logprobs = self.logprobs.astype(np.float32)

        def create_empty_top_data(top_field, dtype):
            if top_field is None or 0 in top_field.shape:
                top_field = np.zeros((self.token_ids.shape[0], self.token_ids.shape[1], 1), dtype=dtype)
            else:
                top_field = top_field.astype(dtype)
            return top_field

        self.top_token_ids = create_empty_top_data(self.top_token_ids, np.int64)
        self.top_logprobs = create_empty_top_data(self.top_logprobs, np.float32)

        if self.cumulative_logprobs is not None:
            self.cumulative_logprobs = self.cumulative_logprobs.astype(np.float32)

    def fill_dummy(self, input_metadata: InputMetadata, max_generated_tokens: int) -> None:
        new_sequence_ids = np.setdiff1d(input_metadata.all_sequence_ids, self.sequence_ids)
        has_running_sequences = len(new_sequence_ids) != len(input_metadata.all_sequence_ids)

        if len(new_sequence_ids) > 0 and has_running_sequences:
            request_slice = []
            batch_sequence_ids = []
            for i, sequence_ids in enumerate(input_metadata.batch_sequence_ids):
                intersected_sequence_ids = np.intersect1d(sequence_ids, new_sequence_ids)
                if len(intersected_sequence_ids) != 0:
                    request_slice.append(i)
                    batch_sequence_ids.append(intersected_sequence_ids)
            new_sequence_ids_group = batch_sequence_ids
        elif not has_running_sequences:
            new_sequence_ids_group = input_metadata.batch_sequence_ids
        else:
            new_sequence_ids_group = []

        for sequence_group in new_sequence_ids_group:
            current_seq_length = self.sequence_ids.size
            self.group_indices.append((current_seq_length, current_seq_length + sequence_group.size))
            self.sequence_ids = np.hstack((self.sequence_ids, sequence_group))
            self.parent_sequence_ids = np.hstack((self.parent_sequence_ids, sequence_group))

        num_new_sequences = sum(len(sg) for sg in new_sequence_ids_group)
        num_padded_tokens = max_generated_tokens - self.token_ids.shape[1]
        pad_width_2d = ((0, num_new_sequences), (0, num_padded_tokens))
        self.token_ids = np.pad(self.token_ids, pad_width_2d, constant_values=-1)
        self.logprobs = np.pad(self.logprobs, pad_width_2d, constant_values=-9999.0)
        pad_width_3d = ((0, num_new_sequences), (0, num_padded_tokens), (0, 0))
        self.top_token_ids = np.pad(self.top_token_ids, pad_width_3d, constant_values=-1)
        self.top_logprobs = np.pad(self.top_logprobs, pad_width_3d, constant_values=-9999.0)
        pad_width_1d = (0, num_new_sequences)
        self.num_new_tokens = np.pad(self.num_new_tokens, pad_width_1d, constant_values=max_generated_tokens)
        self.num_top_tokens = np.pad(self.num_top_tokens, pad_width_1d)
        self.cumulative_logprobs = np.pad(self.cumulative_logprobs, pad_width_1d)
        self.finish_reason = np.pad(self.finish_reason, pad_width_1d)
        self.truncation_indices = np.pad(self.truncation_indices, pad_width_1d)

    def remove(self, removed_sequence_ids: np.ndarray) -> None:
        reserved_indices = []
        new_group_indices = []
        idx = 0
        new_start = 0
        for group_start, group_end in self.group_indices:
            is_reserved = False
            new_end = new_start
            for sequence_id in self.sequence_ids[group_start:group_end]:
                if sequence_id not in removed_sequence_ids:
                    reserved_indices.append(idx)
                    new_end += 1
                    is_reserved = True
                idx += 1
            if is_reserved:
                new_group_indices.append((new_start, new_end))
                new_start = new_end

        self.sequence_ids = self.sequence_ids[reserved_indices]
        self.parent_sequence_ids = self.parent_sequence_ids[reserved_indices]
        self.group_indices = new_group_indices

        fields_to_update = ['token_ids', 'logprobs', 'top_token_ids', 'top_logprobs', 'num_new_tokens',
                            'num_top_tokens', 'cumulative_logprobs', 'finish_reason', 'truncation_indices', 'eos_info']
        for field in fields_to_update:
            setattr(self, field, getattr(self, field)[reserved_indices] if getattr(self, field) is not None else None)

    def pad_output(self, max_generated_tokens):
        num_padded_tokens = max_generated_tokens - self.token_ids.shape[1]
        pad_width_2d = ((0, 0), (0, num_padded_tokens))
        self.token_ids = np.pad(self.token_ids, pad_width_2d, constant_values=-1)
        self.logprobs = np.pad(self.logprobs, pad_width_2d, constant_values=-9999.0)
        pad_width_3d = ((0, 0), (0, num_padded_tokens), (0, 0))
        self.top_token_ids = np.pad(self.top_token_ids, pad_width_3d, constant_values=-1)
        self.top_logprobs = np.pad(self.top_logprobs, pad_width_3d, constant_values=-9999.0)

    def concatenate_output(self, new_output, max_generated_tokens: int) -> None:
        def covert_to_batch_sequence_ids(output):
            tmp_batch_seqence_ids = []
            for start_index, end_index in output.group_indices:
                sequence_ids = output.sequence_ids[start_index:end_index].ravel()
                tmp_batch_seqence_ids.append(sequence_ids)
            return tmp_batch_seqence_ids

        # This function is triggered only when a P-last occurs between D-first and D-last; under normal circumstances,
        # len(new_sequence_ids) == len(new_output.sequence_ids)，has_running_sequences=False
        new_sequence_ids = np.setdiff1d(new_output.sequence_ids, self.sequence_ids)
        has_running_sequences = len(new_sequence_ids) != len(new_output.sequence_ids)

        if len(new_sequence_ids) > 0 and has_running_sequences:
            # Under normal circumstances, this branch will not be entered.
            logger.info("[layerwiseDisaggregated]Two output-wrappers have overlap sequences!")
            request_slice = []
            batch_sequence_ids = []
            # The sequence_ids in new_output are already flattened, and there is no longer a batch_sequence_ids 
            # (their relationship is that the original input is [[1],[2],[3]], while the flattened version is [1,2,3]).
            # However, after the output response, it is necessary to split all sequences back into their corresponding 
            # sequence groups based on group_indices. Therefore, batch_sequence_ids must be reconstructed
            # at this stage; 
            # directly performing an intersection is not feasible.
            seq_idx = 0
            for start_index, end_index in new_output.group_indices:
                sequence_ids = new_output.sequence_ids[start_index:end_index].ravel()
                intersected_sequence_ids = np.intersect1d(sequence_ids, new_sequence_ids)
                if len(intersected_sequence_ids) != 0:
                    request_slice.append(seq_idx)
                    batch_sequence_ids.append(intersected_sequence_ids)
                seq_idx += 1
            new_sequence_ids_group = batch_sequence_ids
        elif not has_running_sequences:
            new_sequence_ids_group = covert_to_batch_sequence_ids(new_output)
        else:
            new_sequence_ids_group = []

        for sequence_group in new_sequence_ids_group:
            current_seq_length = self.sequence_ids.size
            self.group_indices.append((current_seq_length, current_seq_length + sequence_group.size))
            self.sequence_ids = np.hstack((self.sequence_ids, sequence_group))
            self.parent_sequence_ids = np.hstack((self.parent_sequence_ids, sequence_group))
        
        self.pad_output(max_generated_tokens)
        new_output.pad_output(max_generated_tokens)
        
        self.token_ids = np.concatenate([self.token_ids, new_output.token_ids], axis=0)
        self.logprobs = np.concatenate([self.logprobs, new_output.logprobs], axis=0)
        self.top_token_ids = np.concatenate([self.top_token_ids, new_output.top_token_ids], axis=0)
        self.top_logprobs = np.concatenate([self.top_logprobs, new_output.top_logprobs], axis=0)
        # When padding dummy, use constant_values = max_generated_tokens.
        self.num_new_tokens = np.concatenate([self.num_new_tokens, new_output.num_new_tokens], axis=0) 
        self.num_top_tokens = np.concatenate([self.num_top_tokens, new_output.num_top_tokens], axis=0)
        self.cumulative_logprobs = np.concatenate([self.cumulative_logprobs, new_output.cumulative_logprobs], axis=0)
        self.finish_reason = np.concatenate([self.finish_reason, new_output.finish_reason], axis=0)
        self.truncation_indices = np.concatenate([self.truncation_indices, new_output.truncation_indices], axis=0)