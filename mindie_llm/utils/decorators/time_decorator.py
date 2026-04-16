# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import functools
import json
import os
import time
import uuid
import threading

from collections import deque
from threading import Lock, Condition

from ..env import ENV
from ..file_utils import safe_open
from ..log.logging import logger
from ..log.utils import create_log_dir_and_check_permission
from ..tensor import npu

MAX_FILE_SIZE = 10 * 1024 * 1024 * 100


class Timer:
    def __init__(self, logger_instance):
        self.logger = logger_instance
        self.time_cache = {}
        self.cache = deque()  # use deque as cache
        self.cache_lock = Lock()  # thread lock for protect
        self.max_cache_size = 10000  # max items to cache
        self.flush_interval = 30  # unit: second
        self.flush_condition = Condition(self.cache_lock)
        self.flush_thread = threading.Thread(target=self._periodic_flush, daemon=True)
        self.flush_thread.start()

    @staticmethod
    def _write_to_file(cache_to_write):
        reserved_lines = None
        create_log_dir_and_check_permission(ENV.benchmark_filepath)

        if os.path.exists(ENV.benchmark_filepath) and os.path.getsize(ENV.benchmark_filepath) > MAX_FILE_SIZE:
            with safe_open(ENV.benchmark_filepath, "r", encoding="utf-8", max_file_size=2 * MAX_FILE_SIZE) as file:
                lines = file.readlines()
                reserved_lines = lines[-int(ENV.benchmark_reserving_ratio * len(lines)) :]
            os.remove(ENV.benchmark_filepath)

        with safe_open(ENV.benchmark_filepath, "a", encoding="utf-8", max_file_size=2 * MAX_FILE_SIZE) as file:
            if reserved_lines is not None:
                file.writelines(reserved_lines)
            for log_entry in cache_to_write:
                file.write(json.dumps(log_entry) + "\n")

    def track_time(self, logging_name):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if ENV.benchmark_enable:
                    npu.synchronize()
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    npu.synchronize()
                    end_time = time.time()
                    duration = end_time - start_time
                    if self.time_cache.get(logging_name) is None:
                        self.time_cache[logging_name] = [duration]
                    else:
                        self.time_cache[logging_name].append(duration)
                    self.logger.debug(f"{logging_name} took {duration:.6f} seconds to execute")
                    return result
                else:
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def track_time_async(self, logging_name):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if ENV.benchmark_enable_async:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    duration = end_time - start_time
                    if self.time_cache.get(logging_name) is None:
                        self.time_cache[logging_name] = [duration]
                    else:
                        self.time_cache[logging_name].append(duration)
                    self.logger.debug(f"{logging_name} took {duration:.6f} seconds to execute")
                    return result
                else:
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def flush_to_file(self):
        with self.cache_lock:
            if not self.cache:
                return
            cache_to_write = list(self.cache)
            self.cache.clear()
        self._write_to_file(cache_to_write)

    def log_time_async(self, rank, request_ids, token_indices, input_metadata):
        if rank == 0:
            batch_id = str(uuid.uuid4())
            time_info = {k: round(v[-1] * 1000, 4) for k, v in self.time_cache.items()}

            for i, request_id in enumerate(request_ids):
                token_idx = token_indices[i]
                request_time_info = {
                    "is_prefill": input_metadata.is_prefill,
                    "batch_id": batch_id,
                    "request_id": str(request_id),
                    "token_idx": int(token_idx),
                    "unit": "ms",
                }
                request_time_info.update(time_info)

                with self.cache_lock:
                    self.cache.append(request_time_info)

                    # if cache meets the threshold, write data to file
                    if len(self.cache) >= self.max_cache_size:
                        self.flush_condition.notify()

        self.time_cache = {}

    def log_time(self, rank, request_ids, token_indices):
        if rank == 0:
            reserved_lines = None
            create_log_dir_and_check_permission(ENV.benchmark_filepath)
            if os.path.exists(ENV.benchmark_filepath) and os.path.getsize(ENV.benchmark_filepath) > MAX_FILE_SIZE:
                with safe_open(ENV.benchmark_filepath, "r", encoding="utf-8", max_file_size=2 * MAX_FILE_SIZE) as file:
                    lines = file.readlines()
                    reserved_lines = lines[-int(ENV.benchmark_reserving_ratio * len(lines)) :]
                os.remove(ENV.benchmark_filepath)
            batch_id = str(uuid.uuid4())
            time_info = {k: v[-1] * 1000 for k, v in self.time_cache.items()}
            with safe_open(ENV.benchmark_filepath, "a", encoding="utf-8", max_file_size=2 * MAX_FILE_SIZE) as file:
                if reserved_lines is not None:
                    file.writelines(reserved_lines)
                for i, request_id in enumerate(request_ids):
                    token_idx = token_indices[i]
                    request_time_info = {
                        "batch_id": batch_id,
                        "request_id": str(request_id),
                        "token_idx": int(token_idx),
                        "unit": "ms",
                    }
                    request_time_info.update(time_info)
                    file.write(json.dumps(request_time_info) + "\n")
        self.time_cache = {}

    def _periodic_flush(self):
        while True:
            with self.flush_condition:
                if not self.cache:
                    self.flush_condition.wait(timeout=self.flush_interval)
            self.flush_to_file()


timer = Timer(logger)
