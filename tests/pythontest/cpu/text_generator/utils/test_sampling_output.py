# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
import numpy as np

from mindie_llm.text_generator.utils.sampling_output import SamplingOutput


class TestSamplingOutput(unittest.TestCase):
    def setUp(self):
        # 构造一个基础的 SamplingOutput 实例
        # 假设 batch_size=3, max_tokens_generated=5
        self.token_ids = np.array([
            [10, 20, 3, 40, 50],  # 行0: EOS(3) 在索引2
            [11, 21, 31, 41, 51], # 行1: 无 EOS
            [3, 12, 22, 32, 42]   # 行2: EOS(3) 在索引0
        ])
        self.num_new_tokens = np.array([5, 5, 5])
        
        # 初始化 SamplingOutput 对象
        self.sampling_output = SamplingOutput(
            sequence_ids=np.array([0, 1, 2]),
            parent_sequence_ids=np.array([0, 1, 2]),
            group_indices=[(0, 1), (1, 2), (2, 3)],
            repeating_indices=np.zeros(3),
            token_ids=self.token_ids.copy(),
            logprobs=np.zeros((3, 5)),
            top_token_ids=np.zeros((3, 5)),
            top_logprobs=np.zeros((3, 5)),
            cumulative_logprobs=np.zeros(3),
            num_new_tokens=self.num_new_tokens.copy()
        )

    def test_truncate_after_eos(self):
        eos_id = 3
        
        # 执行截断操作
        self.sampling_output.truncate_after_eos(eos_id)

        # 1. 验证行0：EOS在索引2，所以 num_new_tokens 应该是 3，之后的内容应为0
        self.assertEqual(self.sampling_output.num_new_tokens[0], 3)
        np.testing.assert_array_equal(
            self.sampling_output.token_ids[0], 
            np.array([10, 20, 3, 0, 0])
        )

        # 2. 验证行1：没有EOS，数据不应改变
        self.assertEqual(self.sampling_output.num_new_tokens[1], 5)
        np.testing.assert_array_equal(
            self.sampling_output.token_ids[1], 
            np.array([11, 21, 31, 41, 51])
        )

        # 3. 验证行2：EOS在索引0，num_new_tokens 应该是 1
        self.assertEqual(self.sampling_output.num_new_tokens[2], 1)
        np.testing.assert_array_equal(
            self.sampling_output.token_ids[2], 
            np.array([3, 0, 0, 0, 0])
        )

    def test_truncate_with_multiple_eos(self):
        # 测试如果一行中有多个 EOS，是否只在第一个之后截断
        self.sampling_output.token_ids[1] = np.array([10, 3, 3, 3, 10])
        self.sampling_output.num_new_tokens[1] = 5
        
        self.sampling_output.truncate_after_eos(eos_token_id=3)
        
        # 应该在索引1（第一个3）处截断，num_new_tokens变为2
        self.assertEqual(self.sampling_output.num_new_tokens[1], 2)
        np.testing.assert_array_equal(
            self.sampling_output.token_ids[1],
            np.array([10, 3, 0, 0, 0])
        )

if __name__ == '__main__':
    unittest.main()