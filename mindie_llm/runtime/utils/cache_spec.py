# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from dataclasses import dataclass
from enum import IntEnum
import torch


class CacheType(IntEnum):
    TOKEN = 0  # 按token、block_size、ratio去分配block
    SEQUENCE = 1  # 按sequence去分配block，每个sequence固定1个block
    SLIDING_WINDOW = 2  # 按sliding_window分配block，每个sequence固定1个window，每个window大小默认2个block


@dataclass
class CacheSpec:
    dtype: list[torch.dtype]
    format: int
    shape: list[tuple[int, int, int]]
    type: list[CacheType]
    ratio: list[int]


@dataclass
class CacheGroupInfo:
    bytes_of_blocks: int = 0
    num_blocks: int = 1
    block_size: int = 0
    type: CacheType = CacheType.TOKEN
    ratio: int = 1
