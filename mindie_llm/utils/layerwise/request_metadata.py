#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import List
from dataclasses import dataclass, field
from mindie_llm.utils.log.logging import logger


@dataclass
class LwdMetadata:
    # The start and end indices define a left-closed, right-open interval; the first edge is [0, 0],
    # the last edge is [1, 1]; Cloud starts at 0 and spans a total of cloud_total_layer layers.
    request_key: int = 0
    start_exec_layer: int = 0
    end_exec_layer: int = 0
    end_of_generate_token: bool = True
    is_prefill: bool = True
    is_dummy_batch: bool = False
    request_dp_empty: bool = False
    cloud_total_layer: int = 62
    is_long_seq: bool = False
    long_seq_start_idx: int = 0
    long_seq_end_idx: int = 0
    hidden_start_pos: int = 0  # 边侧本段的hidden的起始位置; 云侧使用时是从边侧接收的最后一小段
    prefill_total_seq_len: int = 0
    is_last_chunk: bool = False
    long_seq_recv_list: List[tuple] = field(default_factory=list)  # [(start_position, shape), ...]


class LwdMetadataManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LwdMetadataManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, name=""):
        if not hasattr(self, "initialized"):
            self.name = name
            self.initialized = True

            self.metadata = LwdMetadata()

    def get_metadata(self):
        logger.info(f"[layerwiseDisaggregated] get lwd metadata: {self.metadata}")
        return self.metadata

    def set_metadata(self, metadata: LwdMetadata):
        if metadata is None:
            return

        self.metadata = metadata
        logger.info(f"[layerwiseDisaggregated] set lwd metadata: {metadata}")


lwd_metadata_manager = LwdMetadataManager()
