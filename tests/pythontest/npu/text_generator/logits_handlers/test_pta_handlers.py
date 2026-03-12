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
from unittest.mock import patch, MagicMock

import numpy as np
import torch
import torch_npu

from mindie_llm.text_generator.samplers.logits_handlers.pta_handlers import PTA_HANDLER_REGISTRY
from mindie_llm.text_generator.samplers.sampler_params import HandlerParams
from mindie_llm.text_generator.utils.sampling_metadata import SamplingMetadata


class TestPtaHandlers(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu'
        self.test_logits = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4],
             [-0.1, 0.01, 0.2, 0.02],
             [-0.3, -0.2, -0.1, 0.01]]
        ).to(self.device)
        self.params = HandlerParams(backend_type='atb', rank=0)
        self.params.batch_size = 3
        self.params.vocab_size = 4
        self.params.output_token_ids = torch.tensor(
            [[0, 0, 1, 1, 2, 2, 3, 3],
             [2, 3, 3, 3, 3, 3, 4, 4],
             [4, 4, 4, 4, 4, 4, 4, 4]], dtype=torch.int64
        ).to(self.device)
        self.metadata = SamplingMetadata.from_numpy(
            batch_sequence_ids=[np.array([0]), np.array([1]), np.array([2])],
            is_prefill=False,
            to_tensor=torch.tensor
        )
        self.res_logits = None

    def assert_almost_equal_1d(self, res_list, exp_list):
        torch_npu.npu.synchronize()
        for r, e in zip(res_list, exp_list):
            self.assertAlmostEqual(r, e, places=6, msg=f'\nres:\n{res_list}\nexp:\n{exp_list}\n')

    def assert_almost_equal_2d(self, res_list, exp_list):
        torch_npu.npu.synchronize()
        for res, exp in zip(res_list, exp_list):
            for r, e in zip(res, exp):
                self.assertAlmostEqual(r, e, places=6, msg=f'\nres:\n{res_list}\nexp:\n{exp_list}\n')

    def test_repetition_penalty(self):
        repetition_penalty_tensor = torch.tensor([1.5, 2.5, 0.5]).to(self.device).unsqueeze(1)
        seq_ids_tensor = torch.tensor(
            [[0, 0, 0],
             [1, 1, 1],
             [0, 2, 1]], dtype=torch.int64
        ).to(self.device)
        repetition_penalty_lh = PTA_HANDLER_REGISTRY['repetition_penalty'](self.params)
        self.metadata.repetition_penalty = repetition_penalty_tensor
        self.params.all_token_ids = seq_ids_tensor
        self.res_logits = repetition_penalty_lh(self.test_logits, self.metadata)

    def test_frequency_penalty(self):
        frequency_penalty_tensor = torch.tensor([1.5, 2.5, 0.5]).to(self.device).unsqueeze(1)
        self.metadata.frequency_penalty = frequency_penalty_tensor
        frequency_penalty_lh = PTA_HANDLER_REGISTRY['frequency_penalty'](self.params)
        self.res_logits = frequency_penalty_lh(self.test_logits, self.metadata)

    def test_presence_penalty(self):
        presence_penalty_tensor = torch.tensor([1.5, 2.5, 0.5]).to(self.device).unsqueeze(1)
        self.metadata.presence_penalty = presence_penalty_tensor
        presence_penalty_lh = PTA_HANDLER_REGISTRY['presence_penalty'](self.params)
        self.res_logits = presence_penalty_lh(self.test_logits, self.metadata)

    def test_temperature(self):
        temperature_tensor = torch.tensor([1.5, 1, 0.5]).to(self.device).unsqueeze(1)
        self.metadata.temperature = temperature_tensor
        temperature_lh = PTA_HANDLER_REGISTRY['temperature'](self.params)
        self.res_logits = temperature_lh(self.test_logits, self.metadata)
        expected_logits = [
            [0.1 / 1.5, 0.2 / 1.5, 0.3 / 1.5, 0.4 / 1.5],
            [-0.1 / 1, 0.01 / 1, 0.2 / 1, 0.02 / 1],
            [-0.3 / 0.5, -0.2 / 0.5, -0.1 / 0.5, 0.01 / 0.5]
        ]
        self.assert_almost_equal_2d(self.res_logits.cpu().numpy().tolist(), expected_logits)

    def test_guided_decoding_bitmask_none(self):
        guided_decoding_lh = PTA_HANDLER_REGISTRY['guided_decoding'](self.params)
        self.metadata.guided_bitmask = None
        self.res_logits = guided_decoding_lh(self.test_logits, self.metadata)
        self.assertTrue(torch.equal(self.res_logits, self.test_logits))

    def test_guided_decoding_import_success(self):
        guided_decoding_lh = PTA_HANDLER_REGISTRY['guided_decoding'](self.params)
        bitmask = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.int32)
        self.metadata.guided_bitmask = bitmask
        mock_apply = MagicMock()
        guided_decoding_lh._apply_token_bitmask_inplace = mock_apply
        guided_decoding_lh._import_attempted = True
        self.res_logits = guided_decoding_lh(self.test_logits, self.metadata)
        mock_apply.assert_called_once()

    def test_guided_decoding_import_failure(self):
        guided_decoding_lh = PTA_HANDLER_REGISTRY['guided_decoding'](self.params)
        bitmask = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.int32)
        self.metadata.guided_bitmask = bitmask
        guided_decoding_lh._import_attempted = True
        guided_decoding_lh._apply_token_bitmask_inplace = None
        self.res_logits = guided_decoding_lh(self.test_logits, self.metadata)
        self.assertTrue(torch.equal(self.res_logits, self.test_logits))

    def test_guided_decoding_apply_exception(self):
        guided_decoding_lh = PTA_HANDLER_REGISTRY['guided_decoding'](self.params)
        bitmask = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.int32)
        self.metadata.guided_bitmask = bitmask
        guided_decoding_lh._apply_token_bitmask_inplace = MagicMock(side_effect=RuntimeError("test error"))
        guided_decoding_lh._import_attempted = True
        self.res_logits = guided_decoding_lh(self.test_logits, self.metadata)
        self.assertTrue(torch.equal(self.res_logits, self.test_logits))

    def test_guided_decoding_lazy_import_first_attempt(self):
        guided_decoding_lh = PTA_HANDLER_REGISTRY['guided_decoding'](self.params)
        mock_func = MagicMock()
        with patch.dict('sys.modules', {'mindie_llm.text_generator.plugins.structured_output': MagicMock(apply_token_bitmask_inplace=mock_func)}):
            result = guided_decoding_lh._lazy_import()
            self.assertTrue(guided_decoding_lh._import_attempted)

    def test_guided_decoding_lazy_import_already_attempted(self):
        guided_decoding_lh = PTA_HANDLER_REGISTRY['guided_decoding'](self.params)
        guided_decoding_lh._import_attempted = True
        guided_decoding_lh._apply_token_bitmask_inplace = MagicMock()
        result = guided_decoding_lh._lazy_import()
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()