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


class RequestListener:
    _instance = None

    def __init__(self, config):
        self.communication = None
        self.config = config
        return

    @classmethod
    def get_instance(cls, config):
        if cls._instance is None:
            cls._instance = RequestListener(config)
        return cls._instance

    def start(self):
        from mindie_llm.connector.request_listener.shared_mem_communication import SharedMemCommunication

        self.communication = SharedMemCommunication.get_instance(self.config)
        self.communication.start()
        return True

    def stop(self):
        self.communication.stop()
        return True
