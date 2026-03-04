# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from ...utils.decorators.time_decorator import timer
from ...utils.prof.profiler import span_start, span_end, span_req, span_attr
from ..utils.input_metadata import InputMetadata
from .plugin_manager import PluginManager


class PluginManagerEdge(PluginManager):
    def __init__(self, generator_backend, kvcache_settings, infer_context, output_filter, is_mix_model,
                 plugin_list, **kwargs):
        super().__init__(generator_backend, kvcache_settings, infer_context, output_filter, is_mix_model,
                         plugin_list, **kwargs)
        self.past_key_values = None

    def rollback_kv_cache(self, past_key_values, right_kv_num, q_lens):
        new_past_key_values = []
        if right_kv_num >= 1:
            for layer_key_values in past_key_values:
                new_key_value = []
                key_cache = layer_key_values[0]
                value_cache = layer_key_values[1]
                slot = key_cache.shape[2] - q_lens + right_kv_num
                new_key_value.append(key_cache[:, :, :slot, :])
                new_key_value.append(value_cache[:, :, :slot, :])
                new_past_key_values.append(new_key_value)
        return new_past_key_values

    def deal_kv_cache(self, next_tokens, q_lens, past_key_values):
        decoding_length = 8
        if q_lens is not None and q_lens[0] > 1 and len(next_tokens[0]) < decoding_length:
            past_key_values = self.rollback_kv_cache(past_key_values, len(next_tokens[0]), q_lens[0])
        return past_key_values

    @timer.track_time_async('generate_token')
    def generate_token(self, input_metadata: InputMetadata):
        prof = span_start("preprocess")
        self.plugin_data_param.q_len = None
        self.plugin_data_param.mask = None
        cache_ids, model_inputs, sampling_metadata, trace_ids = self.preprocess(input_metadata)
        model_inputs, qlen, mask = self.model_inputs_update_manager(
            model_inputs, input_metadata, sampling_metadata, cache_ids)
        self.plugin_data_param.q_len = qlen if qlen is not None else self.plugin_data_param.q_len
        self.plugin_data_param.mask = mask if mask is not None else self.plugin_data_param.mask
        span_end(prof)

        prof = span_start("forward", True)
        span_req(prof, trace_ids)
        if hasattr(self.model_wrapper, "mapping"):
            span_attr(prof, "dp_rank", str(self.model_wrapper.mapping.attn_dp.rank))
        logits, past_key_values = self.generator_backend.forward(model_inputs, q_lens=self.plugin_data_param.q_len,
                                                                 spec_mask=self.plugin_data_param.mask,
                                                                 past_key_values=self.past_key_values,
                                                                 soc_version='Ascend310B')

        logits = logits.squeeze(0)
        if model_inputs.is_prefill:
            logits = logits[-1, :].unsqueeze(0)
        span_end(prof, True)

        prof = span_start("sample")
        draft_filtered_logits = self.sample_preprocess_manager(logits, sampling_metadata, input_metadata)
        sampling_output = self.generator_backend.sample(draft_filtered_logits, sampling_metadata)
        span_end(prof)

        prof = span_start("postprocess")
        generation_output = self.postprocess(cache_ids, input_metadata, logits, sampling_metadata, sampling_output)
        self.past_key_values = self.deal_kv_cache(generation_output.token_ids, qlen, past_key_values)
        span_end(prof)

        return generation_output