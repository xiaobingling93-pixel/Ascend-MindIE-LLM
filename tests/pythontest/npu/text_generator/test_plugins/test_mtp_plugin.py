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
from mindie_llm.text_generator.plugins.mtp.decoding_policy import DecodingPolicy
from mindie_llm.text_generator.plugins.mtp.mtp_plugin import MtpPlugin
from mindie_llm.text_generator.utils.model_input import ModelInput
from tests.pythontest.npu import FakeModelRunner, FakeParallelInfo

PLUGIN_PARAMS = '{\"plugin_type\": \"mtp\",\"num_speculative_tokens\":1}'
SPECULATION_GAMMA = 1


class TestMTPPlugin(unittest.TestCase):
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
            'model_instance_type': 'Standard', 'model_name': 'deepseekv3', 'moe_tp': '1',
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
            hidden_states = kwargs.get("hidden_states")
            logits_dim0 = 0
            draft_token = torch.zeros(1, dtype=torch.int64)
            if not model_inputs.is_prefill:
                for i in kwargs.get("q_lens"):
                    logits_dim0 += i
            else:
                logits_dim0 = len(model_inputs.context_length)
            if hidden_states is not None:
                logits_dim0 = SPECULATION_GAMMA
            logits = torch.zeros(logits_dim0, 10) # 假定词表长度为10
            for i in range(logits.shape[0]):
                logits[i][2] = 2
                logits[i][5] = 3
                logits[i][8] = 4
            
            if hidden_states is None:
                hidden_states = torch.rand(logits_dim0, 4096)
            
            return logits, hidden_states, draft_token
        
        mock_model_runner.return_value = self.fake_model_runner
        mock_forward.side_effect = side_effect_forward
        mock_obfuscation_func.return_value = None
        mock_warm_up.return_value = 10

        generator = Generator(self.model_config)

        sample_dtype = generator.infer_context._batch_context.default_sampling_params.dtype
        greedy_param = np.array([(1.0, 0., 0., 0, 1, 1, False, 0)], dtype=sample_dtype)
        input1 = [5159, 636, 374, 31346, 323, 358]
        block_tables = np.array([[0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
        
        gen_len = 20
        req = Request.request_from_token(input1, sampling_params=greedy_param,
                                         generation_params=GenerationParams(max_new_tokens=gen_len))
        meta_data = InputMetadata.from_requests([req], block_tables, True)
        meta_data.batch_block_tables = block_tables
        meta_data.block_rank_id = np.array([4])
        meta_data.is_append_block = np.array([False])
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
            hidden_states = kwargs.get("hidden_states")
            logits_dim0 = 0
            draft_token = torch.zeros(1, dtype=torch.int64)
            if not model_inputs.is_prefill:
                for i in kwargs.get("q_lens"):
                    logits_dim0 += i
            else:
                logits_dim0 = len(model_inputs.context_length)
            if hidden_states is not None:
                logits_dim0 = SPECULATION_GAMMA
            logits = torch.zeros(logits_dim0, 10) # 假定词表长度为10
            for i in range(logits.shape[0]):
                logits[i][2] = 2
                logits[i][5] = 3
                logits[i][8] = 4
            
            if hidden_states is None:
                hidden_states = torch.rand(logits_dim0, 4096)
            
            return logits, hidden_states, draft_token

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
        meta_data.block_rank_id = np.array([4])
        meta_data.is_append_block = np.array([False])
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


class TestMTP(unittest.TestCase):
    @patch('mindie_llm.text_generator.plugins.mtp.decoding_policy.DecodingPolicy',
           return_value=MagicMock(DecodingPolicy))
    def setUp(self, mock):
        self.device = 'npu'
        generator_backend = MagicMock()
        generator_backend.model_wrapper = MagicMock()
        generator_backend.model_wrapper.device = self.device
        generator_backend.rank = 0
        generator_backend.cache_pool = MagicMock()
        kvcache_settings = MagicMock()
        kvcache_settings.dtype = torch.float16
        infer_context = MagicMock()
        infer_context._batch_context = MagicMock()
        infer_context._batch_context.all_ndarray_context = MagicMock()
        block_size = 128
        num_npu_blocks = 3
        self.num_npu_blocks = num_npu_blocks
        self.block_size = block_size

        def get_seq_lens_side_effect(context_handles):
            return infer_context._batch_context.all_ndarray_context.seq_lens[context_handles]
        
        def get_mtp_last_token_num_side_effect(context_handles):
            return infer_context._batch_context.all_ndarray_context.mtp_last_token_num[context_handles]
        
        def get_output_len_count_side_effect(context_handles):
            return infer_context._batch_context.all_ndarray_context.output_len_count[context_handles]
        
        def get_all_input_ids_side_effect(context_handles):
            return infer_context._batch_context.all_ndarray_context.all_input_ids[context_handles]
        
        def block_table_to_slots_side_effect(block_tables):
            return infer_context._batch_context.kv_slots[block_tables]
        
        infer_context.get_seq_lens.side_effect = get_seq_lens_side_effect
        infer_context.get_mtp_last_token_num.side_effect = get_mtp_last_token_num_side_effect
        infer_context.get_output_len_count.side_effect = get_output_len_count_side_effect
        infer_context.get_all_input_ids.side_effect = get_all_input_ids_side_effect
        infer_context.block_table_to_slots.side_effect = block_table_to_slots_side_effect
        infer_context._batch_context.kv_slots = np.arange(num_npu_blocks * block_size).reshape(num_npu_blocks, -1)
        infer_context.spcp_parallel_info.scp_size = 1
        infer_context._batch_context.all_ndarray_context.mtp_last_token_num = np.array([1,1,0,0], dtype=np.int32)
        infer_context._batch_context.all_ndarray_context.seq_lens = np.array([1,2,1,1], dtype=np.int32)
        infer_context._batch_context.all_ndarray_context.all_input_ids = np.full((10,200), 3, dtype=np.int32)
        infer_context._batch_context.all_ndarray_context.output_len_count = np.array([1,2,0,0], dtype=np.int32)
        
        generator_backend.cache_pool.kvcache_settings.num_npu_blocks = num_npu_blocks
        model_role = 'decode'
        self.mtp_plugin = MtpPlugin(
            generator_backend=generator_backend,
            kvcache_settings=kvcache_settings,
            infer_context=infer_context,
            output_filter=MagicMock(),
            plugin_data_param=MagicMock(),
            num_speculative_tokens=1,
            model_role=model_role
        )

        self.mtp_plugin.decoding_policy.num_speculative_tokens = 1
        self.mtp_plugin.decoding_policy.infer_context = infer_context

        def to_tensor(data):
            return torch.tensor(data, device=self.device)

        def to_tensor_async(array):
            host_tensor = torch.from_numpy(array).pin_memory()
            device_tensor = host_tensor.to(self.device, non_blocking=True)
            return device_tensor

        generator_backend.to_tensor = to_tensor
        generator_backend.to_tensor_async = to_tensor_async

    def test_get_mtp_draft_model_inputs_pd_dummy(self):
        mock_hidden_states = torch.randn(2, 768)
        with patch.object(
            self.mtp_plugin.decoding_policy,
            'get_input_hidden_states'
        ) as mock_method:
            mock_method.return_value = mock_hidden_states
            metadata = InputMetadata(
                batch_size=1,
                batch_request_ids=np.array([18446744073709551], dtype=np.int64),
                batch_max_output_lens=np.array([1], dtype=np.int64),
                block_tables=np.array([[self.num_npu_blocks - 1]], dtype=np.int64),
                max_block_size=self.block_size,
                has_sampling=False,
                is_prefill=False,
                input_ids=np.array([]),
                batch_seq_len=np.array([1], dtype=np.int64),
                total_seq_num=1,
                batch_sampling_params=np.array([], dtype=np.float64),
                batch_stop_strings=[],
                batch_stop_token_ids=[],
                computed_blocks=None,
                adapter_ids=[],
                num_npu_blocks=self.num_npu_blocks,
                batch_dp_rank_ids=np.array([0], dtype=np.int64),
                batch_ignore_eos=np.array([]),
                batch_skip_special_tokens=np.array([]),
                batch_include_stop=np.array([]),
                trace_ids=None,
                batch_sequence_ids=[np.array([9223372036854775], dtype=np.int64)],
                batch_best_of=np.array([1]),
                batch_logprobs=np.array([]),
                batch_seeds=np.array([]),
                batch_n=np.array([1]),
                batch_use_beam_search=np.array([False]),
                reserved_sequence_ids=[np.array([])],
                is_dummy_batch=True
            )
            model_inputs = ModelInput(
                input_ids=torch.tensor([1000, 1001]).to(self.device),
                position_ids=torch.tensor([2, 3]).to(self.device),
                block_tables=torch.tensor([[0]]).to(self.device),
                slots=torch.tensor([2, 3]).to(self.device),
                context_length=np.array([4]),
                cached_context_length=np.array([4]),
                max_seq_len=4,
                prefill_head_indices=None,
                is_prefill=False,
                block_tables_array=np.array([[0]]),
                input_lengths=torch.tensor([4]).to(self.device)
            )

            model_inputs_mtp, hidden_states = \
                self.mtp_plugin.decoding_policy.get_mtp_draft_model_inputs(model_inputs, metadata, 0, None)
            self.assertEqual(hidden_states.shape, (2, 768), 
                           f"Expected hidden_states shape (2, 768), but got {hidden_states.shape}")
            expected = np.array([0, 0])
            self.assertTrue(np.array_equal(model_inputs_mtp.input_ids, expected), 
                          f"Expected input_ids {expected}, but got {model_inputs_mtp.input_ids}")

    def test_get_mtp_draft_model_inputs_decode(self):
        mock_hidden_states = torch.randn(2, 768)
        with patch.object(
            self.mtp_plugin.decoding_policy,
            'get_input_hidden_states'
        ) as mock_method:
            mock_method.return_value = mock_hidden_states
            metadata = InputMetadata(
                batch_size=2,
                batch_request_ids=np.array([18446744073709551], dtype=np.int64),
                batch_max_output_lens=np.array([1, 1], dtype=np.int64),
                block_tables=np.array([[0], [0]], dtype=np.int64),
                max_block_size=self.block_size,
                has_sampling=False,
                is_prefill=False,
                input_ids=np.array([]),
                batch_seq_len=np.array([1, 1], dtype=np.int64),
                total_seq_num=1,
                batch_sampling_params=np.array([], dtype=np.float64),
                batch_stop_strings=[],
                batch_stop_token_ids=[],
                computed_blocks=None,
                adapter_ids=[],
                num_npu_blocks=self.num_npu_blocks,
                batch_dp_rank_ids=np.array([0], dtype=np.int64),
                batch_ignore_eos=np.array([]),
                batch_skip_special_tokens=np.array([]),
                batch_include_stop=np.array([]),
                trace_ids=None,
                batch_sequence_ids=[np.array([9223372036854775], dtype=np.int64), np.array([9223372036854773], dtype=np.int64)],
                batch_best_of=np.array([1, 1]),
                batch_logprobs=np.array([]),
                batch_seeds=np.array([]),
                batch_n=np.array([1, 1]),
                batch_use_beam_search=np.array([False]),
                reserved_sequence_ids=[np.array([]), np.array([])],
                is_dummy_batch=False
            )
            model_inputs = ModelInput(
                input_ids=np.array([1000, 1001], dtype=np.int64),
                position_ids=np.array([2, 3], dtype=np.int32),
                block_tables=np.array([[0], [0]], dtype=np.int32),
                slots=np.array([2, 3], dtype=np.int32),
                context_length=np.array([2, 2], dtype=np.int32),
                cached_context_length=np.array([2, 2], dtype=np.int32),
                max_seq_len=4,
                prefill_head_indices=None,
                is_prefill=False,
                block_tables_array=np.array([[0], [0]]),
                input_lengths=torch.tensor([2, 2]).to(self.device)
            )

            model_inputs_mtp, hidden_states = \
                self.mtp_plugin.decoding_policy.get_mtp_draft_model_inputs(model_inputs, metadata,
                                                                           np.array([1, 1], dtype=np.int32), None)
            expected = np.array([3, 0, 3, 0])
            self.assertTrue(np.array_equal(model_inputs_mtp.input_ids, expected),
                          f"Expected input_ids {expected}, but got {model_inputs_mtp.input_ids}")

    def test_get_mtp_draft_model_inputs_first_decode(self):
        mock_hidden_states = torch.randn(2, 768)
        with patch.object(
            self.mtp_plugin.decoding_policy,
            'get_input_hidden_states'
        ) as mock_method:
            mock_method.return_value = mock_hidden_states
            metadata = InputMetadata(
                batch_size=2,
                batch_request_ids=np.array([18446744073709551], dtype=np.int64),
                batch_max_output_lens=np.array([1, 1], dtype=np.int64),
                block_tables=np.array([[0], [0]], dtype=np.int64),
                max_block_size=self.block_size,
                has_sampling=False,
                is_prefill=False,
                input_ids=np.array([]),
                batch_seq_len=np.array([1, 1], dtype=np.int64),
                total_seq_num=1,
                batch_sampling_params=np.array([], dtype=np.float64),
                batch_stop_strings=[],
                batch_stop_token_ids=[],
                computed_blocks=None,
                adapter_ids=[],
                num_npu_blocks=self.num_npu_blocks,
                batch_dp_rank_ids=np.array([0], dtype=np.int64),
                batch_ignore_eos=np.array([]),
                batch_skip_special_tokens=np.array([]),
                batch_include_stop=np.array([]),
                trace_ids=None,
                batch_sequence_ids=[np.array([9223372036854775], dtype=np.int64), np.array([9223372036854773], dtype=np.int64)],
                batch_best_of=np.array([1, 1]),
                batch_logprobs=np.array([]),
                batch_seeds=np.array([]),
                batch_n=np.array([1, 1]),
                batch_use_beam_search=np.array([False]),
                reserved_sequence_ids=[np.array([]), np.array([])],
                is_dummy_batch=False
            )
            model_inputs = ModelInput(
                input_ids=np.array([1000, 1001], dtype=np.int64),
                position_ids=np.array([2, 3], dtype=np.int32),
                block_tables=np.array([[0], [0]], dtype=np.int32),
                slots=np.array([2, 3], dtype=np.int32),
                context_length=np.array([2, 2], dtype=np.int32),
                cached_context_length=np.array([2, 2], dtype=np.int32),
                max_seq_len=4,
                prefill_head_indices=None,
                is_prefill=False,
                block_tables_array=np.array([[0], [0]]),
                input_lengths=torch.tensor([2, 2]).to(self.device)
            )

            model_inputs_mtp, hidden_states = \
                self.mtp_plugin.decoding_policy.get_mtp_draft_model_inputs(model_inputs, metadata,
                                                                           np.array([2, 3], dtype=np.int32), None)
            expected = np.array([3, 0, 3, 0])
            self.assertTrue(np.array_equal(model_inputs_mtp.input_ids, expected),
                          f"Expected input_ids {expected}, but got {model_inputs_mtp.input_ids}")

    def test_get_mtp_draft_model_inputs_with_sp(self):
        mock_hidden_states = torch.randn(2, 768)
        with patch.object(
            self.mtp_plugin.decoding_policy,
            'get_input_hidden_states'
        ) as mock_method:
            mock_method.return_value = mock_hidden_states
            metadata = InputMetadata(
                batch_size=2,
                batch_request_ids=np.array([18446744073709551], dtype=np.int64),
                batch_max_output_lens=np.array([1, 1], dtype=np.int64),
                block_tables=np.array([[0], [0]], dtype=np.int64),
                max_block_size=self.block_size,
                has_sampling=False,
                is_prefill=False,
                input_ids=np.array([]),
                batch_seq_len=np.array([1, 1], dtype=np.int64),
                total_seq_num=1,
                batch_sampling_params=np.array([], dtype=np.float64),
                batch_stop_strings=[],
                batch_stop_token_ids=[],
                computed_blocks=None,
                adapter_ids=[],
                num_npu_blocks=self.num_npu_blocks,
                batch_dp_rank_ids=np.array([0], dtype=np.int64),
                batch_ignore_eos=np.array([]),
                batch_skip_special_tokens=np.array([]),
                batch_include_stop=np.array([]),
                trace_ids=None,
                batch_sequence_ids=[np.array([9223372036854775], dtype=np.int64)],
                batch_best_of=np.array([1, 1]),
                batch_logprobs=np.array([]),
                batch_seeds=np.array([]),
                batch_n=np.array([1, 1]),
                batch_use_beam_search=np.array([False]),
                reserved_sequence_ids=[np.array([])],
                is_dummy_batch=False
            )
            model_inputs = ModelInput(
                input_ids=np.array([1000, 1001], dtype=np.int64),
                position_ids=np.array([2, 3], dtype=np.int32),
                block_tables=np.array([[0], [0]], dtype=np.int32),
                slots=np.array([2, 3], dtype=np.int32),
                context_length=np.array([2, 2], dtype=np.int32),
                cached_context_length=np.array([2, 2], dtype=np.int32),
                max_seq_len=4,
                prefill_head_indices=None,
                is_prefill=False,
                block_tables_array=np.array([[0], [0]]),
                input_lengths=torch.tensor([2, 2]).to(self.device)
            )
            self.mtp_plugin.infer_context.spcp_parallel_info.scp_size = 2
            self.mtp_plugin.decoding_policy.max_block_size = 128
            self.mtp_plugin.decoding_policy.sp_token_and_slot_calc_by_context_length(1, np.array([0, 0]), np.array([[0]]), 2)

            self.mtp_plugin.infer_context.spcp_parallel_info.scp_size = 1

    def test_all_token_ids_padding(self):
        sampling_metadata = MagicMock()
        sampling_metadata.all_token_ids = torch.tensor(np.full((1, 10), 3, dtype=np.int64), device=self.device)
        draft_tokens = np.full((1, 1), 5, dtype=np.int64)
        next_guess_logits_num_per_batch = np.array([2], dtype=np.int32)
        batch_size = 1
        input_ids_pad = self.mtp_plugin.decoding_policy.all_token_ids_padding(sampling_metadata, 
                                                                              next_guess_logits_num_per_batch,
                                                                              batch_size, draft_tokens)
        expected = torch.tensor([
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, -1, -1],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, -1]
        ], device=self.device)

        self.assertTrue(
            torch.equal(input_ids_pad, expected),
            f"Output does not match expected.\nOutput:\n{input_ids_pad.cpu()}\nExpected:\n{expected.cpu()}"
        )

if __name__ == "__main__":
    unittest.main()