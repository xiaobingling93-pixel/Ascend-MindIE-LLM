# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import numpy as np

from .sampling_metadata import SamplingMetadata


class RequestsSamplingCache:
    def __init__(self):
        self.cached_sequence_ids = None
        self.sampling_metadata = None

    def __repr__(self):
        return (
            f"SamplingCache:\n"
            f"cached_sequence_ids: {self.cached_sequence_ids}, "
            f"sampling_metadata: {self.sampling_metadata}."
        )

    def get_from_cache(
        self,
        sequence_ids,
    ):
        if (
            self.cached_sequence_ids is None
            or len(self.cached_sequence_ids) != len(sequence_ids)
            or not np.all(self.cached_sequence_ids == sequence_ids)
        ):
            return None
        else:
            return self.sampling_metadata

    def add_to_cache(
        self,
        sequence_ids,
        sampling_metadata: SamplingMetadata,
    ):
        self.cached_sequence_ids = np.array(sequence_ids, dtype=np.int_)
        self.sampling_metadata = sampling_metadata

    def clear(self):
        self.cached_sequence_ids = None
        self.sampling_metadata = None
