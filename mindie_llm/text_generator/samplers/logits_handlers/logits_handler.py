# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from mindie_llm.utils.tensor import backend
from ..sampler_params import HandlerParams
from ...utils.sampling_metadata import SamplingMetadata


class LogitsHandler:
    def __init__(self, params: HandlerParams):
        self.params = params

    def __call__(self, logits: backend.Tensor, metadata: SamplingMetadata) -> backend.Tensor:
        raise NotImplementedError(f"{self.__class__} is abstract, needed to be implemented.")


class LogitsHandlerList(list):
    def __init__(self):
        super().__init__()

    def __call__(self, logits: backend.Tensor, metadata: SamplingMetadata) -> backend.Tensor:
        for handler in self:
            logits = handler(logits, metadata)
        return logits
