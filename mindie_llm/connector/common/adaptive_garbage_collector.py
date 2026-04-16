# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import threading
import time
import gc
from collections import deque
from mindie_llm.utils.log.logging import logger
from mindie_llm.utils.status import CoreThread


class AdaptiveGarbageCollector:
    """
    monitor service status and perform adaptive garbage collection by adjusting its threshold
    service is busy/idle: increase/decrease threshold to suppress/allow Python gc
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, **kwargs):
        self.running = False
        self.thread = None
        self.original_threshold = gc.get_threshold()
        self.monitor_interval = kwargs.get("monitor_interval", 1)
        # args for gc.set_threshold
        self.gc_threshold_idle = kwargs.get("gc_threshold_idle", [20000, 20, 20])
        self.gc_threshold_busy = kwargs.get("gc_threshold_busy", [50000, 50, 50])
        # monitor the increase of requests
        self._prev_req_count = 0
        self._cur_req_count = 0
        # deque pops front items automatically when len exceeds max size
        self._window_size = kwargs.get("window_size", 8)
        self._sliding_window = deque(maxlen=self._window_size)
        logger.info(f"Adaptive GC initialized. Instance arguments: {kwargs}")

    @classmethod
    def get_instance(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = AdaptiveGarbageCollector(**kwargs)
            return cls._instance

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = CoreThread(target=self._monitor_loop, daemon=True, name="adaptive_garbage_collector")
        self.thread.start()
        logger.info("Adaptive GC started")

    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.thread:
            self.thread.join()
        gc.set_threshold(*self.original_threshold)
        logger.info("Adaptive GC stopped and restored original GC threshold")

    def request_counter_increase(self):
        self._cur_req_count += 1

    def _system_is_busy(self):
        # modify this function according to what you monitor
        is_busy = len(self._sliding_window) and not all(value == 0 for value in self._sliding_window)
        return is_busy

    def _set_gc_when_busy(self):
        gc.set_threshold(*self.gc_threshold_busy)
        logger.debug(f"System is busy, GC is set to a higher threshold: {self.gc_threshold_busy}")

    def _set_gc_when_idle(self):
        gc.set_threshold(*self.gc_threshold_idle)
        logger.debug(f"System is idle, GC is set to a lower threshold: {self.gc_threshold_idle}")

    def _monitor_loop(self):
        while self.running:
            new_req_count = self._cur_req_count - self._prev_req_count
            self._sliding_window.append(new_req_count)
            self._prev_req_count = self._cur_req_count

            if self._system_is_busy():
                self._set_gc_when_busy()
            else:
                self._set_gc_when_idle()
            time.sleep(self.monitor_interval)
