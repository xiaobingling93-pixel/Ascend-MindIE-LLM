# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import random
from functools import wraps

import numpy as np
import torch
import torch_npu
from _cpu_logits_handler import _PostProcessingManager

from .token_selector import TokenSelector
from ...utils.sampling_metadata import SamplingMetadata
from ...utils.sampling_output import SamplingOutput
from ....utils.env import ENV
from ....utils.log.error_code import ErrorCode
from ....utils.log.logging import logger

CPU_SELECTOR_REGISTRY = {}


def register_class(name):
    def decorator(cls):
        CPU_SELECTOR_REGISTRY[name] = cls

        @wraps(cls)
        def wrapper(*args, **kwargs):
            return cls(*args, **kwargs)

        return wrapper

    return decorator


@register_class("top_k_top_p_sampling")
class TopKTopPSamplingTokenSelector(TokenSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = _PostProcessingManager.get_instance(self.params.num_threads, self.params.npu_id)
        self.filter_value = self.params.filter_value
        self.speed_mode, self.use_approx = bool(ENV.speed_mode_type & 2), bool(ENV.speed_mode_type & 1)

    def __call__(self, logits: torch.Tensor, metadata: SamplingMetadata):
        seeds = None
        if metadata.is_prefill and not metadata.is_mix:
            if logits.shape[0] != len(metadata.batch_sequence_ids):
                message = (
                    "The batch size of batch_sequence_ids and logits are inconsistent. This may occur if "
                    "incorrect batch_sequence_ids are provided when calling the sample method of the "
                    "generator_backend or when directly invoking the sampler. Please check the shapes of "
                    "batch_sequence_ids and logits before making these calls."
                )
                logger.error(message, ErrorCode.TEXT_GENERATOR_LOGITS_SHAPE_MISMATCH)
                raise ValueError("The batch size of batch_sequence_ids and logits not equal.")
            self.configure(metadata)
            seeds = metadata.seed_array
        elif logits.shape[0] != len(metadata.all_sequence_ids):
            message = (
                "The batch size of all_sequence_ids and logits are inconsistent. This may occur if incorrect "
                "all_sequence_ids are provided when calling the sample method of the generator_backend or when "
                "directly invoking the sampler. Please check the shapes of all_sequence_ids and logits before "
                "making these calls."
            )
            logger.error(message, ErrorCode.TEXT_GENERATOR_LOGITS_SHAPE_MISMATCH)
            raise ValueError("The batch size of all_sequence_ids and logits not equal.")

        logits, index = self.preprocess_top_k(logits, metadata)
        if metadata.is_prefill and metadata.repeating_indices is not None:
            logits = logits[metadata.repeating_indices]
            index = index[metadata.repeating_indices]
        torch_npu.npu.current_stream().synchronize()
        logits_addr = logits.data_ptr()
        index_addr = index.data_ptr()
        host_dtype = (
            "float32"
            if (logits.dtype == torch.float32)
            else ("bfloat16" if (logits.dtype == torch.bfloat16) else "float16")
        )
        batch_size = len(metadata.all_sequence_ids)
        token_ids_array, logprobs = self.processor.next_token_chooser(
            metadata.all_sequence_ids,
            logits_addr,
            index_addr,
            batch_size,
            logits.shape[1],
            metadata.max_logprobs,
            host_dtype,
            self.speed_mode,
            self.use_approx,
        )
        if self.speed_mode:
            index_array = index.cpu().numpy()
            token_ids_array = index_array[np.arange(index_array.shape[0]).reshape(-1, 1), token_ids_array]

        if metadata.repeating_indices_array is not None:
            num_top_tokens = metadata.num_top_tokens[metadata.repeating_indices_array]
            repeating_indices = metadata.repeating_indices_array
        else:
            num_top_tokens = metadata.num_top_tokens
            repeating_indices = np.arange(len(metadata.all_sequence_ids))
        sampling_output = SamplingOutput(
            group_indices=metadata.group_indices,
            sequence_ids=metadata.all_sequence_ids,
            parent_sequence_ids=metadata.parent_sequence_ids,
            token_ids=token_ids_array[:, 0],
            logprobs=logprobs[:, 0],
            top_token_ids=token_ids_array[:, 1:],
            top_logprobs=logprobs[:, 1:],
            cumulative_logprobs=np.zeros(batch_size, dtype=np.float32),
            repeating_indices=repeating_indices,
            num_new_tokens=np.ones(batch_size, dtype=np.int64),
            num_top_tokens=num_top_tokens,
            seeds=seeds,
        )
        return sampling_output

    def clear(self, sequence_ids: np.ndarray):
        self.processor.delete_configs(sequence_ids)

    def configure(self, metadata: SamplingMetadata):
        if metadata.best_of_array is not None:
            if len(metadata.batch_sequence_ids) != len(metadata.reserved_sequence_ids):
                raise RuntimeError("The size of batch_sequence_ids and reserved_sequence_ids are inconsistent.")

            extended_ids = []
            group_indices = []
            parent_sequence_ids = []
            start = 0
            for i, sequence_ids in enumerate(metadata.batch_sequence_ids):
                extended_ids.append(sequence_ids)
                if len(sequence_ids) != 1:
                    raise ValueError("The size of sequence_ids in prefilling stage is not 1.")
                end = start + 1
                if metadata.best_of_array[i] > 1:
                    num_reserved_ids = metadata.best_of_array[i] - 1
                    extended_ids.append(metadata.reserved_sequence_ids[i][:num_reserved_ids])
                    end += num_reserved_ids
                parent_sequence_ids.extend([sequence_ids[0]] * (end - start))
                group_indices.append((start, end))
                start = end
            metadata.all_sequence_ids = np.concatenate(extended_ids)
            metadata.parent_sequence_ids = np.array(parent_sequence_ids)
            metadata.group_indices = group_indices

            seeds = []
            for i, (best_of) in enumerate(metadata.best_of_array):
                if best_of > 1:
                    rng = random.Random(int(metadata.seed_array[i]))
                    sequence_seeds = [rng.randint(0, 2**63 - 1) for _ in range(best_of)]
                    seeds.extend(sequence_seeds)
                else:
                    seeds.append(metadata.seed_array[i])
            metadata.seed_array = np.array(seeds, dtype=np.int64)

            repeating_indices = np.repeat(np.arange(len(metadata.batch_sequence_ids)), metadata.best_of_array)
            metadata.repeating_indices_array = repeating_indices
            metadata.repeating_indices = metadata.to_tensor(repeating_indices)
            self.processor.set_batch_configs(
                metadata.all_sequence_ids,
                metadata.top_k_array[repeating_indices],
                metadata.top_p_array[repeating_indices],
                metadata.do_sample_array[repeating_indices],
                metadata.top_logprobs_array[repeating_indices],
                metadata.seed_array,
                self.params.sampling_method,
            )
        else:
            self.processor.set_batch_configs(
                metadata.all_sequence_ids,
                metadata.top_k_array,
                metadata.top_p_array,
                metadata.do_sample_array,
                metadata.top_logprobs_array,
                metadata.seed_array,
                self.params.sampling_method,
            )

    def preprocess_top_k(self, logits: torch.Tensor, metadata: SamplingMetadata):
        if metadata.max_top_k > 0:
            k_logits, index = torch.topk(logits, metadata.max_top_k)
            if metadata.top_k_disabled_mask is not None:
                kth_logits = torch.gather(k_logits, 1, metadata.top_k_idx)
                kth_logits.masked_fill_(metadata.top_k_disabled_mask, self.filter_value)
                indices_to_remove = logits < kth_logits
                logits.masked_fill_(indices_to_remove, self.filter_value)
                logits, index = torch.sort(logits, descending=True)
            elif metadata.max_logprobs > metadata.max_top_k:
                logits, index = torch.topk(logits, metadata.max_logprobs)
            else:
                logits = k_logits
        else:
            logits, index = torch.sort(logits, descending=True)
        return logits, index
