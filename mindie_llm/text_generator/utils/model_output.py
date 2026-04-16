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
from typing import Optional, Tuple
from threading import Event
import torch
import numpy as np

from .input_metadata import InputMetadata
from .sampling_metadata import SamplingMetadata
from .sampling_output import SamplingOutput
from ...utils.tensor import BackendTensor


@dataclass
class ModelOutput:
    logits: BackendTensor
    hidden_states: Optional[BackendTensor] = None
    draft_tokens: Optional[BackendTensor] = None
    original_result: Optional[Tuple[BackendTensor, ...]] = None


@dataclass
class ModelOutputWrapper:
    cache_ids: np.ndarray
    input_metadata: InputMetadata
    model_output: ModelOutput
    sampling_metadata: SamplingMetadata
    sampling_output: SamplingOutput
    trace_ids: np.ndarray
    current_dp_sequence_ids: Optional[np.ndarray] = None
    launch_done: Optional[Event] = None
    is_mock: bool = False
    execution_done: torch.npu.Event = None

    @classmethod
    def make_empty(cls):
        empty_wrapper = cls(
            cache_ids=None,
            input_metadata=None,
            model_output=None,
            sampling_metadata=None,
            sampling_output=None,
            trace_ids=None,
            current_dp_sequence_ids=np.array([]),
        )
        return empty_wrapper
