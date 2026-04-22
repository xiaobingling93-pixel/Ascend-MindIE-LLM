# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from abc import ABC, abstractmethod
from typing import Any

from ...utils.sampling_output import SamplingOutput
from ...utils.sampling_metadata import SamplingMetadata


class TokenSelector(ABC):
    def __init__(self, selector_params):
        self.params = selector_params

    @abstractmethod
    def __call__(self, logits: Any, metadata: SamplingMetadata) -> SamplingOutput:
        pass
