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
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from mindie_llm.text_generator.adapter.generator_backend import GeneratorBackend


class TestGeneratorBackend(unittest.TestCase):
    def setUp(self):
        self.device = 'npu'

    @patch("mindie_llm.text_generator.adapter.generator_backend.op.cat", return_value=MagicMock())
    @patch("mindie_llm.text_generator.adapter.generator_backend.GeneratorBackend.__init__", return_value=None)
    def test_warm_up(self, mock_init, mock_cat):
        model_config = {}
        generator_backend = GeneratorBackend(model_config)
        generator_backend.forward = MagicMock(return_value=torch.tensor([[0.1, 0, -0.1],
                                                                         [0.1, 0, -0.1]], device=self.device))
        model_inputs = MagicMock()
        sampling_metadata = MagicMock()
        sampling_metadata.all_sequence_ids = np.array([0, 1])
        sampling_metadata.repetition_penalty = torch.tensor([[0.9]], device=self.device)
        generator_backend.sample = MagicMock()
        generator_backend._warm_up(model_inputs, sampling_metadata=sampling_metadata)


if __name__ == "__main__":
    unittest.main()
