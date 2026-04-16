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

import gc
import time

from mindie_llm.utils.log.logging import logger


class GCMonitor:
    __slots__ = ("_start_time", "total_collections", "total_time", "max_time")

    __instance = None

    def __init__(self):
        self.total_collections = 0
        self.total_time = 0.0
        self.max_time = 0.0
        self._start_time = 0.0
        self._enable_callbacks()

    @staticmethod
    def get_instance():
        if GCMonitor.__instance is None:
            GCMonitor.__instance = GCMonitor()
        return GCMonitor.__instance

    def _enable_callbacks(self):
        gc.callbacks.append(self._callback)

    def _callback(self, phase, info):
        if phase == "start":
            logger.debug(f"GC count:{gc.get_count()}")
            self._start_time = time.perf_counter()
        elif phase == "stop":
            elapsed = time.perf_counter() - self._start_time
            self.total_time += elapsed
            self.total_collections += 1
            self.max_time = max(elapsed, self.max_time)
            logger.debug(f"GC elapsed:{elapsed}")
            for key, value in info.items():
                logger.debug(f"{key}:{value}")
