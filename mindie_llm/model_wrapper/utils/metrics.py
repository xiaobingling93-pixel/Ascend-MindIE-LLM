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

import os
import json
import time
import threading
from mindie_llm.utils.log.logging import logger
from mindie_llm.utils.file_utils import safe_open


METRICS_PATH = "./"
METRIC_TIME_SCALE = 1_000  # 定义常量以表示时间缩放因子
EVENT_NAME = "name"  # 事件名字
EVENT_PH = "ph"  # 事件开始或结束
EVENT_PID = "pid"  # 进程id
EVENT_TID = "tid"  # 线程id
EVENT_TIME_STEP = "ts"  # 微秒级时间戳


def json_file_write(file_path: str, data: list[dict]):
    try:
        with safe_open(file_path, "a", permission_mode=0o640):
            pass
        with safe_open(file_path, "rb+", permission_mode=0o640) as f:
            f.seek(0, 2)
            file_size = f.tell()
            if file_size == 0:
                f.write("[".encode())
            else:
                f.seek(-2, 2)
                f.write(",\n".encode())
            for d in data:
                f.write((json.dumps(d) + ",\n").encode())
            f.seek(-2, 2)
            f.write("]".encode())
    except Exception as e:
        logger.error(f"[METRICS]\t>>> batch metrics output failed, {e}")


class FileMetrics:
    def __init__(self):
        self.pid = os.getpid()
        self.tid = threading.get_ident()
        self.metric_enable = os.getenv("MODEL_WRAPPER_METRICS_OUTPUT_ENABLE", "0") in ("1", "true", "True")
        self.path = os.path.join(os.getenv("MINDIE_LLM_HOME_PATH", METRICS_PATH), "logs", "metrics")
        self.metrics = list()
        self.cursor = 0

        if self.metric_enable:
            logger.info("profiling is enabled.")
            self.ensure_metrics_path()
            threading.Thread(target=self.scheduled_output).start()
        else:
            logger.info("profiling is disenabled.")

    def add_event(self, event_str: str, flag: str, tid: int, details: dict):
        if not self.metric_enable:
            return
        ms_now = str(round(time.time_ns() / METRIC_TIME_SCALE))
        batch_metric = dict()
        batch_metric[EVENT_NAME] = event_str
        batch_metric[EVENT_PH] = flag
        batch_metric[EVENT_PID] = self.pid
        batch_metric[EVENT_TID] = tid
        batch_metric[EVENT_TIME_STEP] = ms_now
        if details["event"] == "generate":
            batch_req_ids = details["batch_req_ids"]
            batch_seq_len = details["batch_seq_len"]
            batch_metric[EVENT_NAME] += f",input_length{[seq_len for seq_len in batch_seq_len]}"
            request_metric = [
                {
                    EVENT_NAME: event_str + f",input_length[{seq_len}]",
                    EVENT_PH: flag,
                    EVENT_PID: self.pid,
                    EVENT_TID: str(req_id),
                    EVENT_TIME_STEP: ms_now,
                }
                for req_id, seq_len in zip(batch_req_ids, batch_seq_len)
            ]
        elif details["event"] == "pullkv":
            batch_req_ids = details["batch_req_ids"]
            batch_seq_len = details["batch_seq_len"]
            get_pull_size = details["get_pull_size"]
            blocks_gb_list = [get_pull_size(seq_len) for seq_len in batch_seq_len]
            batch_metric[EVENT_NAME] += f",total_blocks_Gb[{sum(blocks_gb_list)}]"
            request_metric = [
                {
                    EVENT_NAME: event_str + f",blocks_Gb[{blocks_gb}]",
                    EVENT_PH: flag,
                    EVENT_PID: self.pid,
                    EVENT_TID: str(req_id),
                    EVENT_TIME_STEP: ms_now,
                }
                for req_id, blocks_gb in zip(batch_req_ids, blocks_gb_list)
            ]
        else:
            request_metric = []
        self.metrics.append(batch_metric)
        self.metrics.extend(request_metric)

    def output(self):
        if not self.metric_enable:
            return
        logger.info("output metrics size: %s", len(self.metrics))
        file = self.path + "metrics." + str(self.pid) + ".json"
        try:
            if self.cursor != len(self.metrics):
                begin = self.cursor
                end = len(self.metrics)
                self.cursor = end
                json_file_write(file, self.metrics[begin:end])
        except Exception as e:
            logger.error("[MIE04E13030A] [METRICS]\t>>> output failed, %s", e)

    def ensure_metrics_path(self):
        abs_path = os.path.abspath(self.path)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            logger.info(f"[ensure_metrics_path] create folder '{abs_path}' to save metrics")
        else:
            logger.info(f"[ensure_metrics_path] '{abs_path}' exited.")

    def scheduled_output(self):
        while True:
            time.sleep(60)
            self.output()
