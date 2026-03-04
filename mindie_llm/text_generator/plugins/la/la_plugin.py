# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import numpy as np

from .decoding_policy import DecodingPolicy
from ..plugin import Plugin
from ....utils.env import ENV
from ....utils.tensor import op, tensor_backend
from ....utils.log.logging import logger, print_log


def is_log_enable():
    log_level = ENV.log_file_level
    if log_level == "DEBUG":
        return True
    return False


class LaPlugin(Plugin):
    def __init__(self, generator_backend, kvcache_settings, infer_context, output_filter, plugin_data_param, **kwargs):
        super().__init__()
        self.generator_backend = generator_backend
        self.model_wrapper = self.generator_backend.model_wrapper
        self.kvcache_settings = kvcache_settings
        self.infer_context = infer_context
        self.output_filter = output_filter

        self.plugin_data_param = plugin_data_param
        self.input_metadata = None

        block_size = kvcache_settings.block_size
        self.decoding_policy = DecodingPolicy(kwargs, self.infer_context, self.model_wrapper,
                                              is_log_enable(), block_size, kwargs.get('eos_token_id'))
        self.rank = generator_backend.rank
        level = kwargs.get('level', 4)
        window = kwargs.get('window', 5)
        guess_set_size = kwargs.get("guess_set_size", 5)
        print_log(self.rank, logger.debug, f'Lookahead start, NWG={level}/{window}/{guess_set_size}')

    @staticmethod
    def repeat_sample_param(param_tensor, logits_num_per_batch):
        if param_tensor is None:
            return None
        result = []
        for tensor_tmp, n in zip(param_tensor, logits_num_per_batch):
            repeat_tensor = tensor_tmp.repeat(n, 1)
            result.append(repeat_tensor)
        out_tensor = tensor_backend.cat(result, dim=0)
        return out_tensor

    # LA采样预处理
    def sample_preprocess(self, logits, result, sampling_metadata, input_metadata):
        self.sampling_param = sampling_metadata
        self.input_metadata = input_metadata
        all_sequence_ids = sampling_metadata.all_sequence_ids
        batch_size = input_metadata.batch_size
        if sampling_metadata.is_prefill:
            self.plugin_data_param.q_len = [1] * batch_size
            return logits

        logits_num_per_batch = []

        # decode阶段操作
        ends = self.decoding_policy.cu_seq_len[1:].tolist()
        for batch in range(batch_size):
            guess_token = self.decoding_policy.store_guess_tokens[
                batch] if self.decoding_policy.store_guess_tokens else None
            guess_token_num = 0
            if guess_token is not None:
                guess_token_num = len(guess_token) * len(guess_token[0])
            logits_num_per_batch.append(1 + guess_token_num)

        self.plugin_data_param.q_len = logits_num_per_batch

        total_logits_nums = sum(self.plugin_data_param.q_len)
        next_guess_logits = tensor_backend.zeros(
            (total_logits_nums, tensor_backend.shape(logits, -1)),
            dtype=logits.dtype,
            device=tensor_backend.get_device(logits))

        logits_index = 0
        for batch in range(batch_size):
            req_id = all_sequence_ids[batch]
            guess_token_num = self.plugin_data_param.q_len[batch] - 1
            past_token_len = self.decoding_policy.la_cache.get_past_tokens_len(req_id)
            next_guess_logits[logits_index:logits_index + 1 + guess_token_num] = \
                logits[ends[batch] - past_token_len - guess_token_num - 1:ends[batch] - past_token_len]
            logits_index += (1 + guess_token_num)

        input_ids_pad = self.decoding_policy.get_input_ids_pad(batch_size, sampling_metadata)
        sampling_metadata.all_token_ids = input_ids_pad

        req_id_new = np.concatenate([[req_id] * n for req_id, n in zip(all_sequence_ids, logits_num_per_batch)])
        sampling_metadata.all_sequence_ids = req_id_new
        sampling_metadata.parent_sequence_ids = req_id_new

        if sampling_metadata is not None:
            top_k_idx = self.repeat_sample_param(sampling_metadata.top_k_idx, logits_num_per_batch)
            sampling_metadata.top_k_idx = top_k_idx
            top_k_disabled_mask = self.repeat_sample_param(sampling_metadata.top_k_disabled_mask, logits_num_per_batch)
            sampling_metadata.top_k_disabled_mask = top_k_disabled_mask
            repetition_penalty = self.repeat_sample_param(sampling_metadata.repetition_penalty, logits_num_per_batch)
            sampling_metadata.repetition_penalty = repetition_penalty
            frequency_penalty = self.repeat_sample_param(sampling_metadata.frequency_penalty, logits_num_per_batch)
            sampling_metadata.frequency_penalty = frequency_penalty
            presence_penalty = self.repeat_sample_param(sampling_metadata.presence_penalty, logits_num_per_batch)
            sampling_metadata.presence_penalty = presence_penalty
            temperature = self.repeat_sample_param(sampling_metadata.temperature, logits_num_per_batch)
            sampling_metadata.temperature = temperature

        return next_guess_logits

    def la_token_verify_not_sample(self, input_metadata, next_guess_logits_num_per_batch, next_tokens_uncheck):
        next_tokens_indices = []
        start_pos = 0
        for batch, _ in enumerate(input_metadata.all_sequence_ids):
            if input_metadata.is_prefill:
                next_tokens_indices.append([start_pos])
                start_pos += 1
                continue
            end_pos = start_pos + next_guess_logits_num_per_batch[batch]
            next_guess_indices = list(range(start_pos, end_pos))
            next_guess_by_batch = next_tokens_uncheck[start_pos:end_pos]
            start_pos += next_guess_logits_num_per_batch[batch]

            verify_guess_tokens = None if input_metadata.is_prefill else self.decoding_policy.store_guess_tokens[batch]
            if verify_guess_tokens is None:
                next_tokens_indices.append(next_guess_indices)
                continue

            new_results, need_cal_kv = self.decoding_policy.la_verify_greedy_one_batch(
                verify_guess_tokens, next_guess_by_batch, next_guess_indices)
            if need_cal_kv:
                req_id = input_metadata.all_sequence_ids[batch]
                # la专属
                self.decoding_policy.la_cache.set_need_cal_kv(req_id)
            next_tokens_indices.append(new_results)

        return next_tokens_indices

    def la_cache_update(self, metadata, sampling_output, inp_logits_batch):
        # 由于prefill阶段的inp_logits_batch永远为空，因此argmax仅在decode阶段进行
        inp_tokens_batch = []
        if inp_logits_batch:
            for inp_logits in inp_logits_batch:
                inp_tokens_batch.append(op.argmax(inp_logits, dim=-1).tolist())

        self.decoding_policy.la_update(metadata, sampling_output, inp_tokens_batch)

    # la更新inp_logits_batch的信息，放在cache更新中去，暂时采样先拆出来
    def la_get_inp_logits_batch(self, logits, sampling_metadata):
        inp_logits_batch = []

        if sampling_metadata.is_prefill:
            return inp_logits_batch

        sequence_ids = self.input_metadata.all_sequence_ids
        batch_size = len(sequence_ids)

        ends = self.decoding_policy.cu_seq_len[1:].tolist()
        for batch in range(batch_size):
            req_id = sequence_ids[batch]
            inp_logits_batch.append(self.decoding_policy.get_past_logits(logits, req_id, ends[batch]))

        return inp_logits_batch

    # 第一个插件类函数：输入构造
    def model_inputs_update(self, model_inputs, input_metadata, sampling_metadata, cache_ids, input_len_mask, **kwargs):
        (q_len, attention_mask) = input_len_mask
        model_inputs, q_len, attention_mask = (
            self.decoding_policy.handle_input(model_inputs, input_metadata, attention_mask))

        input_len_mask = (q_len, attention_mask)

        return model_inputs, input_len_mask

    # 第二个插件类函数：token验证
    def plugin_verify(self, sampling_output, cache_ids, result):
        next_tokens_indices = self.la_token_verify_not_sample(
            self.input_metadata, self.plugin_data_param.q_len, sampling_output.token_ids)
        sampling_output.repeating_indices = np.arange(len(cache_ids))
        self.decoding_policy.truncate_token_ids(cache_ids, self.input_metadata, sampling_output, next_tokens_indices)
        self.reshape_speculative_outputs(sampling_output, next_tokens_indices)

    # 第三个插件类函数：LA的cache更新
    def plugin_cache_update(self, cache_ids, sampling_output, la_cache_input, is_prefill=False):
        (logits, sampling_metadata) = la_cache_input

        # inp_tokens_batch, LA分支结果更新
        inp_logits_batch = self.la_get_inp_logits_batch(logits, sampling_metadata)

        # cache更新
        self.la_cache_update(self.input_metadata, sampling_output, inp_logits_batch)

    def plugin_cache_clear(self, cache_ids, finish_reason):
        self.infer_context.last_sampling_metadata.clear()
        for batch, _ in enumerate(self.input_metadata.all_sequence_ids):
            req_id = self.input_metadata.all_sequence_ids[batch]
            if finish_reason[batch] != 0:
                self.decoding_policy.handle_eos(req_id)
            if is_log_enable():
                if finish_reason[batch] != 0:
                    self.decoding_policy.request_stats_list.final_print_req_stats(req_id)