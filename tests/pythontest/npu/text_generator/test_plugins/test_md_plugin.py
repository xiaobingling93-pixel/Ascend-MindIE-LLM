# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import sys
import unittest
from unittest.mock import patch, MagicMock
import torch

import numpy as np

from mindie_llm.text_generator.generator import Generator
from mindie_llm.text_generator.utils.request import Request
from mindie_llm.text_generator.utils.input_metadata import InputMetadata
from mindie_llm.text_generator.adapter.generator_torch import GeneratorTorch
from mindie_llm.text_generator.utils.generation_metadata import GenerationParams

from tests.pythontest.npu import FakeModelRunner, FakeParallelInfo


PLUGIN_PARAMS = '{\"plugin_type\": \"memory_decoding\",\"decoding_length\":16,\"dynamic_algo\":true}'
SPECULATION_GAMMA = 16


class TestMdPlugin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules['_libatb_torch'] = MagicMock()

    @classmethod
    def tearDownClass(cls):
        del sys.modules['_libatb_torch']

    def setUp(self):
        self.model_config = {
            'backend_bin_path': '/usr/local/Ascend/mindie/2.0.RC1/mindie-llm/bin/',
            'backend_modelInstance_id': '0', 'backend_type': 'atb', 'block_size': '128',
            'cpu_mem': '0', 'deploy_type': 'INTER_PROCESS', 'dp': '1', 'executor_type': 'LLM_EXECUTOR_PYTHON',
            'globalRankIds': '', 'globalWorldSize': '0', 'interNodeKmcKsfMaster': 'tools/pmt/master/ksfa',
            'interNodeKmcKsfStandby': 'tools/pmt/standby/ksfb', 'interNodeTLSEnabled': '1',
            'interNodeTlsCaFiles': 'ca.pem,', 'interNodeTlsCaPath': 'security/grpc/ca/',
            'interNodeTlsCert': 'security/grpc/certs/server.pem', 'interNodeTlsCrlFiles': 'server_crl.pem,',
            'interNodeTlsCrlPath': 'security/grpc/certs/', 'interNodeTlsPk': 'security/grpc/keys/server.key.pem',
            'interNodeTlsPkPwd': 'security/grpc/pass/mindie_server_key_pwd.txt', 'isMaster': '0', 'localIP': '',
            'local_rank': '0', 'masterIP': '', 'max_input_len': '2048',
            'max_iter_times': '512', 'max_prefill_tokens': '8192', 'max_seq_len': '2560',
            'model_id': '/home/data/llama3', 'model_instance_number': '1',
            'model_instance_type': 'Standard', 'model_name': 'llama3', 'moe_tp': '1',
            'multiNodesInferEnabled': '0', 'multiNodesInferPort': '1120', 'npu_device_id': '0',
            'npu_device_ids': '0,1,2,3', 'npu_mem': '-1', 'rank': '0', 'slaveIPs': '',
            'tp': '4', 'trust_remote_code': '0', 'world_size': '4',
            'num_speculative_tokens': '0', 'max_batch_size': '5', 'max_prefill_batch_size': '5',
            'distributed_enable': 'false', 'vocab_size': 100000, 'enable_warmup_with_sampling': 'false',
            'cp': '1', 'sp': '1', 'moe_ep': '1'
        }

        plugin_dict = {'plugin_params': PLUGIN_PARAMS, 'speculation_gamma': SPECULATION_GAMMA}
        self.model_config.update(plugin_dict)

        fake_parallel_info = FakeParallelInfo(
            dp=int(self.model_config['dp']),
            tp=int(self.model_config['tp']),
            sp=int(self.model_config['sp']),
            cp=int(self.model_config['cp'])
        )

        self.fake_model_runner = FakeModelRunner(parallel_info=fake_parallel_info)

    @patch('torch.npu.synchronize', return_value=None)
    @patch('atb_llm.runner.model_runner.ModelRunner')
    @patch.object(Generator, 'warm_up')
    @patch.object(GeneratorTorch, '_get_obfuscation_func')
    @patch.object(GeneratorTorch, 'forward')
    def test_generate_token_greedy(
        self,
        mock_forward,
        mock_obfuscation_func,
        mock_warm_up,
        mock_model_runner,
        mock_npu_sync,
    ):

        def side_effect_forward(model_inputs, **kwargs):
            q_lens = 0
            if not model_inputs.is_prefill:
                for i in kwargs.get("q_lens"):
                    q_lens += i
            else:
                q_lens = len(model_inputs.context_length)
            logits = torch.zeros(q_lens, 10) # 假定词表长度为10
            for i in range(q_lens):
                logits[i][2] = 2
                logits[i][5] = 3
                logits[i][8] = 4
            return logits

        mock_model_runner.return_value = self.fake_model_runner

        mock_obfuscation_func.return_value = None
        mock_forward.side_effect = side_effect_forward
        mock_warm_up.return_value = 10

        generator = Generator(self.model_config)

        sample_dtype = generator.infer_context._batch_context.default_sampling_params.dtype
        greedy_param = np.array([(1.0, 0., 0., 0, 1.0, 1.0, False, 0)], dtype=sample_dtype)
        input1 = [5159, 636, 374, 31346, 323, 358]
        block_tables = np.array([[0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
        
        gen_len = 20
        req = Request.request_from_token(input1, sampling_params=greedy_param,
                                         generation_params=GenerationParams(max_new_tokens=gen_len))

        meta_data = InputMetadata.from_requests([req], block_tables, True)
        meta_data.batch_block_tables = block_tables

        generation_output = generator.generate_token(meta_data)

        # 自回归推理
        meta_data.is_prefill = False
        tokens_list = []
        while generation_output.finish_reason[0] == 0:
            generation_output = generator.generate_token(meta_data)
            tokens_list.extend(generation_output.token_ids[0])

        # 验证greedy是否每轮都选择logits最大的token
        is_greedy = True
        for token in tokens_list:
            if token != 8:
                is_greedy = False
                break
        self.assertTrue(is_greedy)


    @patch('torch.npu.synchronize', return_value=None)
    @patch('atb_llm.runner.model_runner.ModelRunner')
    @patch.object(Generator, 'warm_up')
    @patch.object(GeneratorTorch, '_get_obfuscation_func')
    @patch.object(GeneratorTorch, 'forward')
    def test_generate_token_sampling(
        self,
        mock_forward,
        mock_obfuscation_func,
        mock_warm_up,
        mock_model_runner,
        mock_npu_sync,
    ):

        def side_effect_forward(model_inputs, **kwargs):
            q_lens = 0
            if not model_inputs.is_prefill:
                for i in kwargs.get("q_lens"):
                    q_lens += i
            else:
                q_lens = len(model_inputs.context_length)
            logits = torch.zeros(q_lens, 10) # 假定词表长度为10
            for i in range(q_lens):
                logits[i][2] = 2
                logits[i][5] = 3
                logits[i][8] = 4
            return logits

        mock_model_runner.return_value = self.fake_model_runner

        mock_obfuscation_func.return_value = None
        mock_forward.side_effect = side_effect_forward
        mock_warm_up.return_value = 10

        generator = Generator(self.model_config)

        sample_dtype = generator.infer_context._batch_context.default_sampling_params.dtype
        greedy_param = np.array([(1.0, 0., 0., 0.7, 3., 0.92, True, 0)], dtype=sample_dtype)
        input1 = [5159, 636, 374, 31346, 323, 358]
        block_tables = np.array([[0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
        
        gen_len = 20
        req = Request.request_from_token(input1, sampling_params=greedy_param,
                                         generation_params=GenerationParams(max_new_tokens=gen_len))

        meta_data = InputMetadata.from_requests([req], block_tables, True)
        meta_data.batch_block_tables = block_tables

        generation_output = generator.generate_token(meta_data)

        # 自回归推理
        meta_data.is_prefill = False
        tokens_list = []
        while generation_output.finish_reason[0] == 0:
            generation_output = generator.generate_token(meta_data)
            tokens_list.extend(generation_output.token_ids[0])

        # 验证sample是否每轮都选择topk内的token
        is_sampling_valid = True
        valid_token_set = [2, 5, 8]
        for token in tokens_list:
            if token in valid_token_set:
                continue
            else:
                is_sampling_valid = False
                break
        self.assertTrue(is_sampling_valid)

    @patch('atb_llm.runner.model_runner.ModelRunner')
    @patch.object(Generator, 'warm_up')
    @patch.object(GeneratorTorch, '_get_obfuscation_func')
    def test_decoding_length_max(
        self,
        mock_obfuscation_func,
        mock_warm_up,
        mock_model_runner
    ):

        mock_model_runner.return_value = self.fake_model_runner

        mock_obfuscation_func.return_value = None
        mock_warm_up.return_value = 10

        self.model_config["speculation_gamma"] = 17
        self.model_config["plugin_params"] = (
            '{\"plugin_type\": \"memory_decoding\",\"decoding_length\":17,\"dynamic_algo\":true}'
        )
        generator = Generator(self.model_config)
        self.assertEqual(generator.plugin_manager.memory_decoding.decoding_length, 16)

    @patch('atb_llm.runner.model_runner.ModelRunner')
    @patch.object(Generator, 'warm_up')
    @patch.object(GeneratorTorch, '_get_obfuscation_func')
    def test_decoding_length_min(
        self,
        mock_obfuscation_func,
        mock_warm_up,
        mock_model_runner
    ):

        mock_model_runner.return_value = self.fake_model_runner

        mock_obfuscation_func.return_value = None
        mock_warm_up.return_value = 10

        self.model_config["speculation_gamma"] = 17
        self.model_config["plugin_params"] = (
            '{\"plugin_type\": \"memory_decoding\",\"decoding_length\":0,\"dynamic_algo\":true}'
        )
        generator = Generator(self.model_config)
        self.assertEqual(generator.plugin_manager.memory_decoding.decoding_length, 1)

if __name__ == "__main__":
    unittest.main()