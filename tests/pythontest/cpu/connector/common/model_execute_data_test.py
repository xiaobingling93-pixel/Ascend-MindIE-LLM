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

import array
import pytest
import numpy as np

from mindie_llm.connector.common.model_execute_data_pb2 import ExecuteModelRequest, SequenceGroupMetadata, \
    ExecuteResponse


# 测试类
def test_execute_model_request():
    request1 = ExecuteModelRequest()
    metadata = SequenceGroupMetadata()
    s64_array = array.array('q', [1, 3, 4])
    # block_tables is repeated bytes (per block manager)
    metadata.block_tables.append(s64_array.tobytes())
    metadata.stop_token_ids.append(1)
    metadata.stop_token_ids.append(2)
    metadata.stop_token_ids.append(3)

    request1.seq_group_metadata_list.append(metadata)
    # 序列化
    proto_data1 = request1.SerializeToString()

    # 反序列化
    request2 = ExecuteModelRequest()
    request2.ParseFromString(proto_data1)
    metadata2 = request2.seq_group_metadata_list[0]
    blocks = np.frombuffer(metadata2.block_tables[0], dtype=np.int64).tolist()
    assert blocks[0] == 1
    assert blocks[1] == 3
    assert blocks[2] == 4
    print(metadata.sampling_params.repetition_penalty)
    # 测试 ExecuteModelRequest


def test_execute_response():
    resp_type = 0
    ret = 0
    init_results = {'status': "ok"}
    py_response_list = []
    proto_response = ExecuteResponse()
    proto_response.msg_type = resp_type
    proto_response.status = ret
    for key, value in init_results.items():
        proto_response.init_results.init_result_map[key] = value
    # npuBlockNum is now carried in kv_cache_descs
    desc0 = proto_response.init_results.kv_cache_descs.add()
    desc0.npu_block_num = 5727
    desc0.block_size = 128
    desc0.compression_ratio = 1
    desc0.cache_type = 0
    for py_response in py_response_list:
        sequence_output = py_response.parse_from_numpy_array()
        proto_response.execute_model_response.sequence_output.append(sequence_output)
    assert proto_response.init_results.kv_cache_descs[0].npu_block_num == 5727


if __name__ == '__main__':
    pytest.main()
