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

import unittest

import numpy as np

from mindie_llm.text_generator.utils.generation_output import GenerationOutput


class TestGenerationOutput(unittest.TestCase):
    def setUp(self):
        self.generation_output = GenerationOutput(
            sequence_ids=np.array([0, 1, 2]),
            parent_sequence_ids=np.array([0, 0, 0]),
            group_indices=[(0, 1), (1, 2), (2, 3)],
            token_ids=np.array([np.array([100]), np.array([101]), np.array([102])]),
            logprobs=np.array([np.array([-0.1]), np.array([-0.2]), np.array([-0.3])]),
            top_token_ids=np.array([[[100]], [[101]], [[102]]]),
            top_logprobs=np.array([[[-0.1]], [[-0.2]], [[-0.3]]]),
            num_new_tokens=np.array([1, 1, 1]),
            num_top_tokens=np.array([1, 1, 1]),
            cumulative_logprobs=np.array([-0.1, -0.2, -0.3]),
            finish_reason=np.array([0, 0, 0]),
            truncation_indices=np.array([0, 0, 0]),
            current_token_indices=[1, 1, 1],
            trace_ids=np.array([0, 1, 2])
        )

    def test_collate(self):
        self.generation_output.collate()
        self.assertEqual(self.generation_output.eos_info.tolist(), [[0, 1], [0, 1], [0, 1]])

    def test_pad_output(self):
        max_generated_tokens = 3
        self.generation_output.pad_output(max_generated_tokens)
        self.assertEqual(self.generation_output.token_ids.shape, (3, 3))

    def _assert_concatenated_attrs(self, original, new, concatenated, axis=0):
        expected = np.concatenate([original, new], axis=axis)
        np.testing.assert_array_equal(concatenated, expected)

    def test_concatenate_normal_no_overlap(self):

        self.base_new_output = GenerationOutput(
            sequence_ids=np.array([3, 4, 5]),
            parent_sequence_ids=np.array([3, 3, 3]),
            group_indices=[(0, 1), (1, 2), (2, 3)],
            token_ids=np.array([[200], [201], [202]]),
            logprobs=np.array([[-0.4], [-0.5], [-0.6]]),
            top_token_ids=np.array([[[200]], [[201]], [[202]]]),
            top_logprobs=np.array([[[-0.4]], [[-0.5]], [[-0.6]]]),
            num_new_tokens=np.array([1, 1, 1]),
            num_top_tokens=np.array([1, 1, 1]),
            cumulative_logprobs=np.array([-0.4, -0.5, -0.6]),
            finish_reason=np.array([1, 1, 1]),
            truncation_indices=np.array([1, 1, 1]),
            current_token_indices=[2, 2, 2],
            trace_ids=np.array([3, 4, 5])
        )

        max_generated_tokens = 2
        new_output = self.base_new_output

        self.generation_output.concatenate_output(new_output, max_generated_tokens)

        expected_seq_ids = np.array([0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(self.generation_output.sequence_ids, expected_seq_ids)

        expected_group_indices = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
        self.assertEqual(self.generation_output.group_indices, expected_group_indices)

    def test_make_empty(self):
        empty_output = GenerationOutput.make_empty()
        self.assertEqual(empty_output.sequence_ids.shape, (0,))
        self.assertEqual(empty_output.group_indices, [])
        self.assertEqual(empty_output.token_ids.shape, (0, 0))
        self.assertEqual(empty_output.eos_info.shape, (0,))

    def test_remove(self):
        self.generation_output.collate()
        self.generation_output.remove(np.array([1]))
        np.testing.assert_array_equal(self.generation_output.sequence_ids, np.array([0, 2]))
        self.assertEqual(self.generation_output.group_indices, [(0, 1), (1, 2)])
        self.assertEqual(self.generation_output.token_ids.shape[0], 2)
        self.assertEqual(self.generation_output.eos_info.shape[0], 2)


if __name__ == '__main__':
    unittest.main()
