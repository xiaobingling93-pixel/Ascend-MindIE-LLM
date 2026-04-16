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


class ProcessStartArgName:
    LOCAL_RANK = "local_rank"
    LOCAL_WORLD_SIZE = "local_world_size"
    GLOBAL_RANK = "global_rank"
    GLOBAL_WORLD_SIZE = "global_world_size"
    NPU_NUM_PER_DP = "npu_num_per_dp"
    NPU_DEVICE_ID = "npu_device_id"
    PARENT_PID = "parent_pid"
    SHM_NAME_PREFIX = "shm_name_prefix"
    SEM_NAME_PREFIX = "sem_name_prefix"
    COMMUNICATION_TYPE = "communication_type"
    USE_MOCK_MODEL = "use_mock_model"
    LAYERWISE_DISAGGREGATED = "layerwise_disaggregated"
    LAYERWISE_DISAGGREGATED_ROLE_TYPE = "layerwise_disaggregated_role_type"
