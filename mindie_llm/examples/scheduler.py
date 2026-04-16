#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import math
import copy
from typing import List

import numpy as np

from mindie_llm.text_generator.utils.request import Request, Sequence
from mindie_llm.utils.tensor import backend, op


def decode_token(req_list: List[Request], tokenizer):
    decode_res_list = []
    token_num_list = []
    cumulative_logprobs = []
    logprobs_list = []
    top_logprobs_list = []
    request_id = 0
    token_num = 0
    for req in req_list:
        for seq in req.completed:
            out_token = len(seq.out_token_list)
            truncation_idx = seq.truncation_indices
            seq.out_token_list = [token_id for token_id in seq.out_token_list if token_id != -1]
            token_tensor = backend.tensor(seq.out_token_list, dtype=backend.int64)
            if tokenizer:
                decode_text = tokenizer.decode(token_tensor, skip_special_tokens=req.skip_special_tokens)
                if truncation_idx != 0:
                    decode_text = decode_text[:truncation_idx]
                decode_res_list.append(decode_text)
            else:
                decode_res_list.append(token_tensor)
            token_num += out_token
            token_num_list.append((request_id, token_num))
            cumulative_logprobs.append(seq.cumulative_logprobs)
            logprobs_list.append(seq.logprobs)
            top_logprobs_list.append(seq.top_logprobs)
            request_id += 1
    return decode_res_list, token_num_list, cumulative_logprobs, logprobs_list, top_logprobs_list


class Scheduler:
    def __init__(
        self, max_batch_size, max_prefill_tokens, generator, load_tokenizer, is_mix_model, split_chunk_tokens, **kwargs
    ):
        self.max_batch_size = max_batch_size
        self.max_prefill_tokens = max_prefill_tokens

        self.generator = generator
        self.tokenizer = self.generator.model_wrapper.tokenizer if load_tokenizer else None
        self.kvcache_settings = self.generator.kvcache_settings
        self.free_block_mask = op.ones(self.kvcache_settings.num_npu_blocks, dtype=backend.int64)

        self.is_mix_model = is_mix_model
        self.split_chunk_tokens = split_chunk_tokens
        self.speculation_gamma = kwargs.get("speculation_gamma", 0)

        self.sequence_id_counter = -1

    @staticmethod
    def get_decoding_requests(requests):
        decoding_requests = []
        for request in requests:
            finished_ids = []
            for seq_id, sequence in request.sequences.items():
                if sequence.eos_flag:
                    request.completed.append(sequence)
                    finished_ids.append(seq_id)
            for seq_id in finished_ids:
                request.sequences.pop(seq_id)
            if request.sequences:
                decoding_requests.append(request)
            if request.use_beam_search:
                completed = sorted(request.completed, key=lambda x: x.cumulative_logprobs, reverse=True)
                request.completed = completed[: request.n]
        return decoding_requests

    @staticmethod
    def get_status_by_batch_req_status(batch_req_status):
        is_prefill_batch = np.where(batch_req_status != 2, True, False)
        return is_prefill_batch

    @staticmethod
    def is_mix_batch_judge(batch_req_status):
        batch_size = len(batch_req_status)
        decode_bsnum = np.count_nonzero(batch_req_status == 2)
        last_prefill_bsnum = np.count_nonzero(batch_req_status == 1)
        if (decode_bsnum == batch_size) or (last_prefill_bsnum == batch_size):
            return False
        else:
            return True

    @staticmethod
    def _collate_sequence_outputs(sequence: Sequence, generation_output, idx):
        sequence.out_token_list.extend(generation_output.token_ids[idx][: generation_output.eos_info[idx][1]])
        sequence.logprobs.extend(generation_output.logprobs[idx][: generation_output.eos_info[idx][1]])
        sequence.eos_flag = generation_output.eos_info[idx][0]
        sequence.kv_length += generation_output.eos_info[idx][1]
        sequence.truncation_indices = generation_output.truncation_indices[idx]
        for i in range(generation_output.eos_info[idx][1]):
            top_logprobs_ins = {}
            num_top_tokens = 0
            if generation_output.num_top_tokens is not None:
                num_top_tokens = generation_output.num_top_tokens[idx]
            for j, top_token_id in enumerate(generation_output.top_token_ids[idx][i][:num_top_tokens]):
                top_logprobs_ins[top_token_id] = generation_output.top_logprobs[idx][i][j]
            sequence.top_logprobs.append(top_logprobs_ins)
            if generation_output.cumulative_logprobs is not None:
                sequence.cumulative_logprobs = generation_output.cumulative_logprobs[idx]

    def can_batch(self, batch_size, tokens_num, input_tokens_num):
        if batch_size >= self.max_batch_size:
            return False
        if self.max_prefill_tokens and tokens_num > self.max_prefill_tokens:
            return False
        if self.is_mix_model and input_tokens_num >= self.split_chunk_tokens:
            return False
        return True

    def get_sequence_id(self):
        self.sequence_id_counter += 1
        return self.sequence_id_counter

    def get_prefilling_requests(self, requests, req_begin_idx=0):
        batch_size = 0
        tokens_num = 0
        prefill_requests = []
        block_tables = []
        begin_idx = min(req_begin_idx, len(requests) - 1)
        end_idx = req_begin_idx
        free_block_idx = 0
        block_len_max = 0
        for i in range(begin_idx, len(requests)):
            if self.can_batch(batch_size, tokens_num, 0):
                req = requests[i]

                batch_size += 1
                max_seq_len = req.max_new_tokens + req.input_length + self.speculation_gamma
                tokens_num += max_seq_len
                if self.kvcache_settings.block_size == 0:
                    raise ValueError("self.kvcache_settings.block_size should not be 0.")
                req.num_sequence_blocks = math.ceil(
                    (req.input_length + req.max_new_tokens) / self.kvcache_settings.block_size
                )
                if req.use_beam_search:
                    req.reserved_seq_ids = [self.get_sequence_id() for _ in range(req.n - 1)]
                    curr_need_blocks = req.num_sequence_blocks * req.n
                else:
                    req.reserved_seq_ids = [self.get_sequence_id() for _ in range(req.best_of - 1)]
                    curr_need_blocks = req.num_sequence_blocks * req.best_of

                req.block_tables = np.arange(free_block_idx, free_block_idx + curr_need_blocks, dtype=np.int32)
                default_seq = list(req.sequences.values())[0]
                default_seq.block_tables = req.block_tables[: req.num_sequence_blocks]
                default_seq.kv_length = req.input_length
                block_len_max = len(req.block_tables) if len(req.block_tables) > block_len_max else block_len_max
                free_block_idx += curr_need_blocks
                block_tables.append(req.block_tables)
                prefill_requests.append(req)
                end_idx = i
            else:
                break

        return prefill_requests, end_idx

    def generate(self, requests):
        if self.is_mix_model:
            return self.generate_is_mix_model(requests)
        output_requests = []
        req_begin_idx = 0
        while req_begin_idx < len(requests):
            prefilling_requests, end_idx = self.get_prefilling_requests(requests, req_begin_idx)
            req_begin_idx = end_idx + 1

            if prefilling_requests:
                generation_output = self.generator.prefill(prefilling_requests)
                self.parse_outputs(prefilling_requests, generation_output, True)
                decoding_requests = self.get_decoding_requests(prefilling_requests)

                while decoding_requests:
                    generation_output = self.generator.decode(decoding_requests)
                    self.parse_outputs(decoding_requests, generation_output, False)
                    decoding_requests = self.get_decoding_requests(decoding_requests)
            output_requests.extend(prefilling_requests)

        return decode_token(output_requests, self.tokenizer)

    def generate_is_mix_model(self, requests):
        stop_generate = False
        req_begin_idx = 0
        truncation_indices_list = []

        while req_begin_idx < len(requests):
            # step1: 首轮prefill切块以及对应的block_table生成, 同时输出batch_req_status用于记录每一个batch的req状态：
            # 0：prefill切块，1：最后一个prefill切块，2：decode
            prefill_requests, batch_req_status, end_idx, alias_input_ids_list = self.get_first_spf_prefill_requests(
                requests, req_begin_idx
            )

            # step2: prefill_requests存在, 则进行第一轮推理
            if prefill_requests:
                is_prefill_batch = self.get_status_by_batch_req_status(batch_req_status)
                request_tokens_np, eof_np, stop_generate, truncation_indices = self.generator.generate_mix(
                    prefill_requests, is_prefill_batch
                )
                for i, request in enumerate(prefill_requests):
                    if batch_req_status[i] == 1:
                        request.out_token_list.extend(list(request_tokens_np[i]))
                truncation_indices_list.append(truncation_indices)

            # step3: mix阶段判断, 始终刷新prefill_requests、batch_req_status
            mix_status = self.is_mix_batch_judge(batch_req_status)
            while mix_status and (not stop_generate):
                prefill_requests, batch_req_status = self.get_spf_requests(
                    prefill_requests, batch_req_status, alias_input_ids_list
                )
                is_prefill_batch = self.get_status_by_batch_req_status(batch_req_status)
                request_tokens_np, eof_np, stop_generate, truncation_indices = self.generator.generate_mix(
                    prefill_requests, is_prefill_batch
                )
                for i, request in enumerate(prefill_requests):
                    if batch_req_status[i] == 1:
                        request.out_token_list.extend(list(request_tokens_np[i]))
                    elif batch_req_status[i] == 2:
                        idx = eof_np[i][1]
                        out_tokens = request_tokens_np[i][:idx]
                        request.out_token_list.extend(out_tokens)
                truncation_indices_list[-1] = truncation_indices
                mix_status = self.is_mix_batch_judge(batch_req_status)

            # step4: 判断是否进入纯decode阶段
            while not stop_generate:
                is_prefill_batch = [0] * len(is_prefill_batch)
                request_tokens_np, eof_np, stop_generate, truncation_indices = self.generator.generate_mix(
                    prefill_requests, is_prefill_batch
                )
                for i, request in enumerate(prefill_requests):
                    idx = eof_np[i][1]
                    out_tokens = request_tokens_np[i][:idx]
                    request.out_token_list.extend(out_tokens)
                truncation_indices_list[-1] = truncation_indices
            req_begin_idx = end_idx + 1

        # step5: 生成输出list
        return decode_token(requests, self.tokenizer, truncation_indices_list)

    def get_first_spf_prefill_requests(self, requests, req_begin_idx):
        batch_size = 0
        tokens_num = 0
        input_tokens_num = 0
        prefill_requests = []
        block_tables = []
        batch_req_status = []
        begin_idx = min(req_begin_idx, len(requests) - 1)
        end_idx = req_begin_idx
        free_block_idx = 0
        block_len_max = 0
        alias_input_ids_list = []
        for i in range(begin_idx, len(requests)):
            if self.can_batch(batch_size, tokens_num, input_tokens_num):
                req = requests[i]
                batch_size += 1
                max_seq_len = req.max_new_tokens + req.input_length + self.speculation_gamma
                tokens_num += max_seq_len
                if self.kvcache_settings.block_size == 0:
                    raise ValueError("self.kvcache_settings.block_size should not be 0.")
                curr_need_blocks = math.ceil(max_seq_len / self.kvcache_settings.block_size)
                req.block_tables = np.arange(free_block_idx, free_block_idx + curr_need_blocks + 1, dtype=np.int32)
                block_len_max = len(req.block_tables) if len(req.block_tables) > block_len_max else block_len_max
                free_block_idx += curr_need_blocks + 1
                block_tables.append(req.block_tables)
                alias_input_ids_list.append(requests[i].input_ids)
                if self.is_not_last_prompt(input_tokens_num, req.input_length):
                    req.split_start_position = 0
                    req.split_end_position = self.split_chunk_tokens - input_tokens_num
                    req.input_ids = req.input_ids[req.split_start_position : req.split_end_position]
                    req.last_prompt = 0
                    status = 0
                else:
                    req.split_start_position = 0
                    req.split_end_position = req.input_length
                    req.last_prompt = 1
                    status = 1
                input_tokens_num += req.split_end_position - req.split_start_position
                batch_req_status.append(status)
                prefill_requests.append(req)
                end_idx = i
            else:
                break
        batch_req_status = np.array(batch_req_status, dtype=np.int32)
        batch_result = (prefill_requests, batch_req_status, end_idx, alias_input_ids_list)
        return batch_result

    def get_spf_requests(self, requests, batch_req_status, alias_input_ids_list):
        tokens_num = 0
        for i, req in enumerate(requests):
            if batch_req_status[i] == 0:  # 上一次没有处理到最后一个prefill块
                prefill_len_unprocess = len(alias_input_ids_list[i]) - req.split_end_position
                if self.is_not_last_prompt(tokens_num, prefill_len_unprocess):
                    req.split_start_position = req.split_end_position
                    req.split_end_position = req.split_start_position + self.split_chunk_tokens - tokens_num
                    req.last_prompt = 0
                else:
                    req.split_start_position = req.split_end_position
                    req.split_end_position = len(alias_input_ids_list[i])
                    req.last_prompt = 1
                    batch_req_status[i] = 1
            elif batch_req_status[i] == 1:  # 上一次是最后一个prefill快
                req.split_start_position = 0
                req.split_end_position = 1
                req.last_prompt = 1
                batch_req_status[i] = 2
            req.input_ids = alias_input_ids_list[i][req.split_start_position : req.split_end_position]
            tokens_num += req.split_end_position - req.split_start_position
            requests[i] = req
        return requests, batch_req_status

    def parse_outputs(self, requests, generation_output, is_prefill):
        src_dst_indices = []
        idx = 0
        for i, req in enumerate(requests):
            forking_idx = 0
            group_start, group_end = generation_output.group_indices[i]
            request_sequence_ids = generation_output.sequence_ids[group_start:group_end]
            request_parent_ids = generation_output.parent_sequence_ids[group_start:group_end]

            discarded_seq_ids = []
            for pre_seq_id in req.sequences.keys():
                if pre_seq_id not in request_sequence_ids:
                    discarded_seq_ids.append(pre_seq_id)

            reused_block_id = 0
            for j, seq_id in enumerate(request_sequence_ids):
                parent_id = request_parent_ids[j]
                if seq_id != -1 and seq_id != parent_id:
                    # fork block tables
                    req.reserved_seq_ids.remove(seq_id)
                    if req.use_beam_search:
                        req.reserved_seq_ids.append(self.get_sequence_id())
                    req.sequences[seq_id] = copy.deepcopy(req.sequences[parent_id])
                    req.sequences[seq_id].seq_id = seq_id

                    src_index = math.floor(req.sequences[parent_id].kv_length / self.kvcache_settings.block_size)
                    need_forking = True
                    if not req.sequences[parent_id].kv_length % self.kvcache_settings.block_size:
                        src_index = src_index + 1
                        need_forking = False
                    dst_step = req.num_sequence_blocks - src_index

                    if is_prefill:
                        forking_idx += 1
                        dst_index = src_index + forking_idx * dst_step
                        if need_forking:
                            src_dst_indices.append((req.block_tables[src_index], req.block_tables[dst_index]))
                        req.sequences[seq_id].block_tables = np.concatenate(
                            [
                                req.sequences[parent_id].block_tables[:src_index],
                                req.block_tables[dst_index : dst_index + dst_step],
                            ]
                        )
                    else:
                        dis_id = discarded_seq_ids[reused_block_id]
                        reused_block_tables = req.sequences[dis_id].block_tables
                        reused_block_id += 1
                        if need_forking:
                            src_dst_indices.append(
                                (req.sequences[parent_id].block_tables[src_index], reused_block_tables[src_index])
                            )
                        req.sequences[seq_id].block_tables = np.concatenate(
                            [
                                req.sequences[parent_id].block_tables[:src_index],
                                reused_block_tables[src_index : req.num_sequence_blocks],
                            ]
                        )

            for j, seq_id in enumerate(request_sequence_ids):
                parent_id = request_parent_ids[j]
                if seq_id == -1:
                    sequence = copy.deepcopy(req.sequences[parent_id])
                    self._collate_sequence_outputs(sequence, generation_output, idx)
                    req.completed.append(sequence)
                else:
                    self._collate_sequence_outputs(req.sequences[seq_id], generation_output, idx)
                idx += 1

            for discarded in discarded_seq_ids:
                req.sequences.pop(discarded)

        src_dst_indices = np.asarray(src_dst_indices)
        if len(src_dst_indices):
            self.generator.copy_blocks(src_dst_indices)

    def is_not_last_prompt(self, tokens_num, left_token_num):
        if self.is_mix_model is not True:
            return False
        if left_token_num + tokens_num <= self.split_chunk_tokens:
            return False
        return True
