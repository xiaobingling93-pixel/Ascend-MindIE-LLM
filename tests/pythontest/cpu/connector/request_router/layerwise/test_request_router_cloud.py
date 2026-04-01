# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import multiprocessing
import queue
import json
import time
import unittest
from unittest.mock import Mock, MagicMock, patch

from mindie_llm.connector.common.model_execute_data_pb2 import ExecuteRequest, ExecuteType, ForwardType, ExecuteResponse
from mindie_llm.connector.request_router.layerwise.request_router_cloud import RequestRouterCloud
from mindie_llm.utils.layerwise.share_memory import SharedMemoryManager
from mindie_llm.utils.layerwise.request_metadata import LwdMetadata, lwd_metadata_manager
from mindie_llm.connector.request_router.router_impl import RouterImpl
from mindie_llm.utils.layerwise.communication import LwdCommunicationManager
from mindie_llm.connector.request_listener.shared_mem_communication import SharedMemCommunication
from mindie_llm.connector.request_router.layerwise.request_router_lwd import REQUEST_KEY_DECODE, REQUEST_KEY_PREFILL, DecisionType, LastExecType, ModelType
from mindie_llm.connector.request_router.layerwise.request_router_lwd import RequestInfo


class TestRequestRouterCloud(unittest.TestCase):
    def setUp(self):
        self.router = RequestRouterCloud('0')
        self.router.router_impl = Mock(spec=RouterImpl)
        edge_cloud_comm = Mock(spec=LwdCommunicationManager)
        self.router.ctrl_comm = edge_cloud_comm.ctrl_comm
        self.router.data_comm = edge_cloud_comm.data_comm
        self.router.rank = 0
        self.router.mem_manager = self.get_shared_memery()

    @classmethod
    def setUpClass(cls):
        print("TestRequestRouterCloud start")

    @classmethod
    def tearDownClass(cls):
        print("TestRequestRouterCloud end")

    def test_init(self):
        self.assertIsInstance(self.router.inference_queue, queue.Queue)
        self.assertIsInstance(self.router.transfer_queue, queue.Queue)
        self.assertIsInstance(self.router.pdlink_queue, queue.Queue)
        self.assertIsInstance(self.router.prefill_queue, queue.Queue)
        self.assertIsInstance(self.router.decode_queue, queue.Queue)
        self.assertIsInstance(self.router.clean_up_queue, queue.Queue)
        self.assertTrue(self.router.inference_related_thread.is_alive())
        self.assertTrue(self.router.trans_related_thread.is_alive())
        self.assertIsNone(self.router.enable_dp_distributed)

    def get_shared_memery(self):
        share_mem_manager = SharedMemoryManager('0')
        is_producer = True if self.router.rank == 0 else False
        share_mem_manager.initialize(is_producer, 7)   # 8 cards 1 master

        return share_mem_manager

    def mock_prefill_request(self) -> ExecuteRequest:
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_INFER
        mock_execute_model_request = Mock()
        mock_execute_model_request.forward_type = ForwardType.PREFILL
        mock_request.execute_model_request = mock_execute_model_request
        return mock_request
    
    def mock_decode_request(self) -> ExecuteRequest:
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_INFER
        mock_execute_model_request = Mock()
        mock_execute_model_request.forward_type = ForwardType.DECODE
        mock_request.execute_model_request = mock_execute_model_request
        return mock_request
    
    def mock_init_request(self) -> ExecuteRequest:
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_INIT
        mock_request.config = True
        return mock_request
    

    def mock_finalize_request(self) -> ExecuteRequest:
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_FINALIZE
        return mock_request

    def mock_generator_cleanup_request(self) -> ExecuteRequest:
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.TEXT_GENERATOR_CLEANUP
        return mock_request

    def mock_generator_clean_eos_request(self) -> ExecuteRequest:
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.EOS_CLEANUP
        return mock_request
        

    
    def test_adjust_prefill_cut_policy(self):
        self.router.total_layer_num = 64
        self.router.start_layer_num = 1    
        self.router.end_layer_num = 1      

        expect_cut_policy = [
            [31, 31],
            [21, 21, 20],
            [16, 16, 15, 15],
            [13, 13, 12, 12, 12],
            [11, 11, 10, 10, 10, 10],
            [9, 9, 9, 9, 9, 9, 8],
            [8, 8, 8, 8, 8, 8, 7, 7],
        ]
        for i, cut_num in enumerate(range(2,9)):
            prefill_layers_divi_policy = self.router.prepare_prefill_cut_policy(cut_num)
            self.assertEqual(prefill_layers_divi_policy, expect_cut_policy[i])

    def test_prepare_prefill_chunk_policy(self):
        prefill_chunk_policy = self.router.prepare_prefill_chunk_policy(20000, 6)
        self.assertEqual(prefill_chunk_policy, [3334, 3334, 3333, 3333, 3333, 3333])

    def test_save_inference_request(self):
        self.router.initialize = Mock()
        self.router.router_impl.seq_ctrl = Mock()
        self.router.save_inference_request(self.mock_prefill_request())
        self.router.save_inference_request(self.mock_decode_request())
        self.router.save_inference_request(self.mock_init_request())
        self.router.save_inference_request(self.mock_finalize_request())
        self.router.save_inference_request(self.mock_generator_cleanup_request())
        self.assertEqual(self.router.prefill_queue.qsize(), 1)
        self.assertEqual(self.router.decode_queue.qsize(), 1)
        self.assertEqual(self.router.clean_up_queue.qsize(), 1)
    
    def test_get_all_request(self):
        self.router.initialize = Mock()
        self.router.router_impl.seq_ctrl = Mock()
        self.router.ctrl_comm.prefill_comm_finish_tcp_count = 0
        self.router.data_comm.p_shape = [0]
        self.router.data_comm.recv_index = 0
        
        self.router.parse_all_dp_batches_seq_lens = MagicMock(return_value=[0])
        self.router.calc_max_seq_len = MagicMock(return_value=0)
        self.router.calc_curr_dp_seq_len = MagicMock(return_value=0)
        self.router.calc_batch_size = MagicMock(return_value=1)
        self.router.prefill_layers_divi_switch = False
        
        self.router.prefill_chunk_instance = Mock()
        self.router.prefill_chunk_instance.model_type = ModelType.QWEN
        
        self.router.inference_queue.put(self.mock_prefill_request())
        self.router.inference_queue.put(self.mock_decode_request())
        self.router.inference_queue.put(self.mock_init_request())
        self.router.inference_queue.put(self.mock_finalize_request())
        self.router.inference_queue.put(self.mock_generator_cleanup_request())
        self.router.get_all_request()
        self.assertIsNotNone(self.router.request_map[REQUEST_KEY_PREFILL])
        self.assertIsNotNone(self.router.request_map[REQUEST_KEY_DECODE])

    def test_calc_decision_type_dd_other(self):
        self.router.clean_up_queue.put("other_request")
        self.router.prefill_request = None
        self.router.decode_request = None
        self.router.prefill_queue = queue.Queue()
        self.router.decode_queue = queue.Queue()
        decision_type = self.router.calc_decision_type()
        self.assertEqual(decision_type, DecisionType.DO_CLEAN_UP)

    def test_calc_decision_type_do_prefille(self):
        mock_metadata = Mock()
        self.router.prefill_metadata_queue = [(0, mock_metadata, None)]
        self.router.decode_metadata_queue = [(0, mock_metadata, None)]
        self.router.prefill_comm_finish = True
        self.router.decode_comm_finish = True
        self.router.last_execute_type = LastExecType.DECODE
        decision_type = self.router.calc_decision_type()
        self.assertEqual(decision_type, DecisionType.DO_PREFILL)

    def test_calc_decision_type_do_decode(self):
        mock_decode_metadata = Mock()
        self.router.prefill_metadata_queue = []
        self.router.decode_metadata_queue = [(0, mock_decode_metadata, None)]
        self.router.decode_comm_finish = True
        decision_type = self.router.calc_decision_type()
        self.assertEqual(decision_type, DecisionType.DO_DECODE) 

    def test_calc_decision_type_wait_decode(self):
        mock_metadata = Mock()
        self.router.prefill_metadata_queue = [(0, mock_metadata, None)]
        self.router.decode_metadata_queue = []
        self.router.prefill_comm_finish = True
        self.router.last_execute_type = LastExecType.PREFILL
        self.router.before_last_execute_type = LastExecType.DECODE
        self.router.prefill_exec_last_time = time.time()
        decision_type = self.router.calc_decision_type()
        self.assertEqual(decision_type, DecisionType.WAIT_DECODE)

    def test_calc_decision_type_do_prefille_after_wait_decode(self):
        mock_metadata = Mock()
        self.router.prefill_metadata_queue = [(0, mock_metadata, None)]
        self.router.decode_metadata_queue = []
        self.router.prefill_comm_finish = True
        self.router.last_execute_type = LastExecType.PREFILL
        self.router.before_last_execute_type = LastExecType.DECODE
        self.router.prefill_exec_last_time = time.time() - 0.02  # 0.02 秒前
        decision_type = self.router.calc_decision_type()
        self.assertEqual(decision_type, DecisionType.DO_PREFILL)

    def test_calc_decision_type_wait_comm(self):
        self.router.prefill_metadata_queue = []
        self.router.decode_metadata_queue = []
        self.router.prefill_comm_finish = False
        self.router.decode_comm_finish = False
        decision_type = self.router.calc_decision_type()
        self.assertEqual(decision_type, DecisionType.WAIT_COMM)

    def test_calc_decision_type_clean_eos(self):
        self.router.prefill_metadata_queue = []
        self.router.decode_metadata_queue = []
        self.router.prefill_comm_finish = False
        self.router.decode_comm_finish = False
        self.router.clean_eos_queue.put(self.mock_generator_clean_eos_request())
        decision_type = self.router.calc_decision_type()
        self.assertEqual(decision_type, DecisionType.DO_CLEAN_EOS)

    def test_recv_prefill(self):
        mock_metadata = Mock()
        long_seq_decision_meta = MagicMock()
        long_seq_decision_meta.chunk_index_size = 1
        self.router.prefill_metadata_queue = [(0, mock_metadata, long_seq_decision_meta)]
        self.router.decode_metadata_queue = []
        self.router.prefill_comm_finish = False
        self.router.ctrl_comm.recv_prefill = Mock()
        self.router.ctrl_comm.prefill_comm_finish_tcp_count = 1
        self.router.ctrl_comm.prefill_recv_msg = None

        self.router.prefill_comm_finish = False
        self.router.recv_prefill()
        self.assertTrue(self.router.prefill_comm_finish)

    def test_recv_decode(self):
        self.router.decode_comm_finish = False
        self.router.ctrl_comm.recv_decode = Mock()
        self.router.recv_decode()
        self.assertTrue(self.router.decode_comm_finish) 

    def test_arrange_exec_stage_do_decode(self):
        # 准备测试数据
        request_key = 0
        metadata = LwdMetadata(request_key, 0, 62, True, False, False, False, 62, False, 0, 0, 0, 0, False)
        self.router.decode_metadata_queue.append((request_key, metadata))
        # 调用方法
        self.router.arrange_exec_stage(DecisionType.DO_DECODE)
        metadata_ = lwd_metadata_manager.get_metadata()
        test_cases = [(0, 62, True, 62, False, False, 0, 0)]
        for (start_exec_layer, end_exec_layer, end_of_generate_token, cloud_total_layer, is_prefill, is_long_seq, long_seq_start_idx, long_seq_end_idx) in test_cases:
            self.assertEqual(metadata_.start_exec_layer, start_exec_layer)
            self.assertEqual(metadata_.end_exec_layer, end_exec_layer)
            self.assertEqual(metadata_.end_of_generate_token, end_of_generate_token)
            self.assertEqual(metadata_.cloud_total_layer, cloud_total_layer)
            self.assertEqual(metadata_.is_prefill, is_prefill)
            self.assertEqual(metadata_.is_long_seq, is_long_seq)
            self.assertEqual(metadata_.long_seq_start_idx, long_seq_start_idx)
            self.assertEqual(metadata_.long_seq_end_idx, long_seq_end_idx)
            

    def test_arrange_exec_stage_do_prefille_no_chunk(self):
        self.router.prefill_layers_divi_policy = [13, 13, 12, 12, 12]
        self.router.is_long_seq = False
        self.router.prefill_exec_start_layer = 0
        self.router.prefill_exec_cnt = 0
        test_cases = [
            (0, 13, False, 62, True, False, 0, 0),
            (13, 26, False, 62, True, False, 0, 0),
            (26, 38, False, 62, True, False, 0, 0),
            (38, 50, False, 62, True, False, 0, 0),
            (50, 62, True, 62, True, False, 0, 0)
        ]

        for i, case in enumerate(test_cases):
            self.router.prefill_exec_cnt = i
            # 准备测试数据
            request_key = 0
            metadata = LwdMetadata(request_key, case[0], case[1], case[2], 
                                  case[4], False, False, case[3], case[5], 
                                  case[6], case[7], 0, 0, False)
            self.router.prefill_metadata_queue.append((request_key, metadata, None))
            # 调用方法
            self.router.arrange_exec_stage(DecisionType.DO_PREFILL)
            metadata_ = lwd_metadata_manager.get_metadata()
            (start_exec_layer, end_exec_layer, end_of_generate_token, cloud_total_layer, is_prefill, is_long_seq, long_seq_start_idx, long_seq_end_idx) = test_cases[i]
            self.assertEqual(metadata_.start_exec_layer, start_exec_layer, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.end_exec_layer, end_exec_layer, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.end_of_generate_token, end_of_generate_token, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.cloud_total_layer, cloud_total_layer, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.is_prefill, is_prefill, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.is_long_seq, is_long_seq, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.long_seq_start_idx, long_seq_start_idx, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.long_seq_end_idx, long_seq_end_idx, f"Failed on test case {i+1}")


    def test_arrange_exec_stage_do_prefille_with_chunk(self):
        self.router.prefill_layers_divi_switch = False
        self.router.prefill_layers_divi_policy = [8, 8, 8, 8, 8, 8, 7, 7]
        self.router.prefill_layers_divi_num = len(self.router.prefill_layers_divi_policy)
        self.router.is_long_seq = True
        self.router.prefill_chunk_policy = [25000, 25000, 25000, 25000]
        self.router.prefill_chunk_num = len(self.router.prefill_chunk_policy)
        self.router.prefill_seq_len = 100000
        self.router.prefill_exec_start_layer = 0
        self.router.prefill_exec_cnt = 0
        self.router.long_seq_start_idx = 0
        self.router.prefill_exec_chunk_cnt = 0

        test_cases = [
            (0, 8, False, 62, True, True, 0, 25000),
            (8, 16, False, 62, True, True, 0, 25000),
            (16, 24, False, 62, True, True, 0, 25000),
            (24, 32, False, 62, True, True, 0, 25000),
            (32, 40, False, 62, True, True, 0, 25000),
            (40, 48, False, 62, True, True, 0, 25000),
            (48, 55, False, 62, True, True, 0, 25000),
            (55, 62, False, 62, True, True, 0, 25000),
            (0, 8, False, 62, True, True, 25000, 50000),
            (8, 16, False, 62, True, True, 25000, 50000),
            (16, 24, False, 62, True, True, 25000, 50000),
            (24, 32, False, 62, True, True, 25000, 50000),
            (32, 40, False, 62, True, True, 25000, 50000),
            (40, 48, False, 62, True, True, 25000, 50000),
            (48, 55, False, 62, True, True, 25000, 50000),
            (55, 62, False, 62, True, True, 25000, 50000),
            (0, 8, False, 62, True, True, 50000, 75000),
            (8, 16, False, 62, True, True, 50000, 75000),
            (16, 24, False, 62, True, True, 50000, 75000),
            (24, 32, False, 62, True, True, 50000, 75000),
            (32, 40, False, 62, True, True, 50000, 75000),
            (40, 48, False, 62, True, True, 50000, 75000),
            (48, 55, False, 62, True, True, 50000, 75000),
            (55, 62, False, 62, True, True, 50000, 75000),
            (0, 8, False, 62, True, True, 75000, 100000),
            (8, 16, False, 62, True, True, 75000, 100000),
            (16, 24, False, 62, True, True, 75000, 100000),
            (24, 32, False, 62, True, True, 75000, 100000),
            (32, 40, False, 62, True, True, 75000, 100000),
            (40, 48, False, 62, True, True, 75000, 100000),
            (48, 55, False, 62, True, True, 75000, 100000),
            (55, 62, True, 62, True, True, 75000, 100000),
        ]
        for i, case in enumerate(test_cases):
            # 准备测试数据
            request_key = 0
            metadata = LwdMetadata(request_key, case[0], case[1], case[2], 
                                  case[4], False, False, case[3], case[5], 
                                  case[6], case[7], 0, 0, False)
            self.router.prefill_metadata_queue.append((request_key, metadata, None))
            # 调用方法
            self.router.arrange_exec_stage(DecisionType.DO_PREFILL)
            metadata_ = lwd_metadata_manager.get_metadata()
            (start_exec_layer, end_exec_layer, end_of_generate_token, cloud_total_layer, is_prefill, is_long_seq, long_seq_start_idx, long_seq_end_idx) = test_cases[i]
            self.assertEqual(metadata_.start_exec_layer, start_exec_layer, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.end_exec_layer, end_exec_layer, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.end_of_generate_token, end_of_generate_token, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.cloud_total_layer, cloud_total_layer, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.is_prefill, is_prefill, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.is_long_seq, is_long_seq, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.long_seq_start_idx, long_seq_start_idx, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.long_seq_end_idx, long_seq_end_idx, f"Failed on test case {i+1}")


    def test_shared_memory(self):
        request_key = 0
        self.router.request_map[REQUEST_KEY_PREFILL][request_key] = RequestInfo(request=ExecuteRequest())
        self.router.request_map[REQUEST_KEY_DECODE][request_key] = RequestInfo(request=ExecuteRequest())
        self.router.prefill_chunk_instance = Mock()
        self.router.prefill_chunk_instance.model_type = ModelType.QWEN
        
        self.router.rank = 0
        self.router.broadcast_decision_type(DecisionType.DO_PREFILL, request_key)
        for rank in range(1, 8):
            self.router.rank = rank
            decision_type = self.router.recv_decision_type()
            self.assertEqual(decision_type, DecisionType.DO_PREFILL)

        self.router.rank = 0
        decision_type = self.router.broadcast_decision_type(DecisionType.DO_DECODE, request_key)

        for rank in range(1, 8):
            self.router.rank = rank
            decision_type = self.router.recv_decision_type()
            self.assertEqual(decision_type, DecisionType.DO_DECODE)

    
    @patch.object(SharedMemoryManager, 'initialize')
    @patch('mindie_llm.connector.common.send_model_execute_response')
    @patch('mindie_llm.connector.request_listener.shared_mem_communication.SharedMemCommunication.send_model_execute_response_cls')
    @patch.object(LwdCommunicationManager, 'initialize')
    @patch.object(LwdCommunicationManager, 'communication_config_verify', return_value=True)
    @patch('mindie_llm.connector.request_router.layerwise.request_router_lwd.ExecuteResponseBuilder')
    @patch('mindie_llm.connector.request_router.request_router.RouterImpl')
    @patch("mindie_llm.connector.request_router.request_router.BaseConfig")
    def test_initialize_standard_mode(self, mock_base_config, mock_router_impl, mock_response_builder, *args):
        mock_config = Mock()
        mock_config.items.return_value = [
            ("infer_mode", "standard"),
            ("cpu_mem", "2048"),
            ("model_path", "/path/to/standard/model"),
            ("device", "npu"),
            ("distributed_enable", True),
            ("local_rank", "0"),
            ("model_id", "0"),
            ("rank", "0"),
            ("world_size", "2"),
            ("npu_device_id", "0"),
            ("npu_device_ids", "0,1"),
            ("cpu_mem", "0"),
            ("npu_mem", "-1"),
            ("max_seq_len", "133000"),
            ("max_iter_times", "136096"),
            ("max_prefill_tokens", "133000"),
            ("max_input_len", "133000"),
            ("block_size", "128"),
            ("layerwiseDisaggregated", "true"),
            ("layerwiseDisaggregatedRoleType", "slave"),
            ("edgeIpAddress", "127.0.0.0"),
            ("cloudIpAddress", "127.0.0.0"),
            ("max_iter_times", "136096"),
            ("max_batch_size", "200"),
            ("max_prefill_tokens", "136096"),
            ("trust_remote_code", "false"),
            ("backend_type", "atb"),
            ("models", json.dumps({"startLayerNum": 1}))
        ]

        mock_base_config_instance = Mock()
        mock_model_config = dict(mock_config.items.return_value)
        mock_base_config_instance.model_config = mock_model_config
        mock_base_config.return_value = mock_base_config_instance

        mock_initialize_result = {"status": "ok"}
        mock_router_impl.initialize.return_value = mock_initialize_result

        mock_response = Mock(spec=ExecuteResponse)
        mock_response_builder.build_from_init_result.return_value = mock_response

        # 模拟 router_impl 和 edge_cloud_comm
        self.router.router_impl = Mock(spec=RouterImpl)
        self.router.ctrl_comm = Mock()
        self.router.data_comm = Mock()
        self.router.prepare_prefill_cut_policy = Mock()

        router_impl = Mock()
        router_impl.generator.model_wrapper = Mock()
        router_impl.generator.model_wrapper.model_runner = Mock()
        router_impl.generator.model_wrapper.model_runner.config = Mock()
        router_impl.generator.model_wrapper.model_runner.config.num_hidden_layers = 64
        mock_router_impl.return_value = router_impl

        # 模拟 SharedMemCommunication._instance
        mock_shared_mem_comm_instance = Mock()
        mock_shared_mem_comm_instance.send_response = Mock()
        SharedMemCommunication._instance = mock_shared_mem_comm_instance

        self.router.initialize(mock_config)

    def test_do_clean_up(self):
        self.router.initialize = Mock()
        self.router.router_impl.finalize = Mock()
        self.router.router_impl.seq_ctrl = Mock()

        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.TEXT_GENERATOR_CLEANUP
        self.router.clean_up_queue.put(mock_request)
        self.router.do_clean_up()
        self.router.router_impl.seq_ctrl.assert_called_once()

    def test_do_clean_eos(self):
        self.router.initialize = Mock()
        self.router.router_impl.finalize = Mock()
        self.router.router_impl.seq_ctrl = Mock()

        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.EOS_CLEANUP
        self.router.clean_eos_queue.put(mock_request)
        self.router.do_clean_eos()
        self.router.router_impl.seq_ctrl.assert_called_once()

    def test_do_prefill(self):
        request_key = 0
        self.router.request_map[REQUEST_KEY_PREFILL][request_key] = RequestInfo(request=ExecuteRequest())
        # 模拟metadata队列
        metadata = LwdMetadata(request_key, 0, 13, False, True, False, False, 62, False, 0, 0, 0, 0, False)
        self.router.prefill_metadata_queue.append((request_key, metadata, None))
        # 调用方法
        self.router.execute_inference_request(DecisionType.DO_PREFILL, request_key)
        self.assertIsNone(self.router.request_map[REQUEST_KEY_PREFILL][request_key])

    def test_do_decode(self):
        request_key = 0
        self.router.request_map[REQUEST_KEY_DECODE][request_key] = RequestInfo(request=ExecuteRequest())
        # 模拟metadata队列
        metadata = LwdMetadata(request_key, 0, 62, True, False, False, False, 62, False, 0, 0, 0, 0, False)
        self.router.decode_metadata_queue.append((request_key, metadata))
        # 调用方法
        self.router.execute_inference_request(DecisionType.DO_DECODE, request_key)
        self.assertIsNone(self.router.request_map[REQUEST_KEY_DECODE][request_key])

    def test_do_inference(self):
        self.router.get_all_request = Mock()
        self.router.recv_prefill = Mock()
        self.router.recv_decode = Mock()
        self.router.calc_decision_type = Mock(return_value=DecisionType.WAIT_COMM)
        self.router.arrange_exec_stage = Mock(return_value=0)
        self.router.broadcast_decision_type = Mock()
        self.router.recv_decision_type = Mock(return_value=DecisionType.WAIT_COMM)
        self.router.execute_inference_request = Mock()
        self.router.rank = 0
        self.router.comm_initialized = True
        self.router.curr_no_request = lambda: False

        timeout_seconds = 0.5
        process = multiprocessing.Process(target=self.router.do_inference)
        process.start()
        process.join(timeout_seconds)
        process.terminate()

        self.router.comm_initialized = True
        self.router.rank = 0
        request_key = 0
        self.router.request_map[REQUEST_KEY_PREFILL][request_key] = RequestInfo(request=ExecuteRequest())
        process = multiprocessing.Process(target=self.router.do_inference)
        process.start()
        process.join(timeout_seconds)
        process.terminate()

        self.router.rank = 1 
        self.router.comm_initialized = True
        process = multiprocessing.Process(target=self.router.do_inference)
        process.start()
        process.join(timeout_seconds)
        process.terminate()
    
    def test_accept(self):
        self.router.calc_seq_len = Mock()
        self.router.calc_seq_len.return_value = 1024
        self.router.calc_batch_size = Mock()
        self.router.calc_batch_size.return_value = 200
        self.router.data_comm.lock = MagicMock()
        self.router.get_prefill_gap_time_list = Mock()
        self.router.prefill_layers_divi_switch = False
        self.router.initialize = Mock()
        self.router.isqwenvl = False
        self.router.data_comm.prefill_seq_len_queue = queue.Queue()
        self.router.data_comm.decode_batch_size_queue = queue.Queue()
        self.router.prefill_chunk_instance = Mock()
        self.router.prefill_chunk_instance.model_type = ModelType.QWEN
        
        self.router.parse_all_dp_batches_seq_lens = MagicMock(return_value=[0])
        self.router.calc_max_seq_len = MagicMock(return_value=0)
        self.router.calc_curr_dp_seq_len = MagicMock(return_value=0)
        self.router.calc_batch_size = MagicMock(return_value=1)
        self.router.ctrl_comm.prefill_comm_finish_tcp_count = 0
        self.router.data_comm.p_shape = [0]
        self.router.data_comm.recv_index = 0
        self.router.calc_curr_dp_batch_size = MagicMock(return_value=0)
        
        self.router.accept(self.mock_prefill_request())
        self.router.accept(self.mock_decode_request())
        self.router.accept(self.mock_init_request())
        self.router.accept(self.mock_finalize_request())
        self.router.accept(self.mock_generator_cleanup_request())

    def test_do_other_request_now(self):
        execute_request = self.mock_prefill_request()
        with self.assertRaises(RuntimeError) as _:
            self.router.do_other_request_now(execute_request)

    def test_update_prefill_layers_divi_num_with_switch_off(self):
        # 测试prefill_layers_divi_switch为False的情况
        self.router.prefill_layers_divi_switch = False
        self.router.cloud_layer_num = 62
        
        # 模拟request_map
        request_key = 0
        mock_request_info = Mock()
        mock_request_info.request = self.mock_prefill_request()
        self.router.request_map[REQUEST_KEY_PREFILL] = {}
        self.router.request_map[REQUEST_KEY_PREFILL][request_key] = mock_request_info
        
        # 调用方法
        prefill_layers_divi_policy = self.router.update_prefill_layers_divi_num(request_key)
        
        # 验证结果
        expected_policy = [13, 13, 12, 12, 12]  # 62层切5份
        self.assertEqual(prefill_layers_divi_policy, expected_policy)

    def test_update_prefill_layers_divi_num_with_switch_on(self):
        # 测试prefill_layers_divi_switch为True的情况
        self.router.prefill_layers_divi_switch = True
        self.router.cloud_layer_num = 62
        
        # 模拟request_map
        request_key = 0
        mock_request_info = Mock()
        mock_request = self.mock_prefill_request()
        mock_request.execute_model_request.seq_group_metadata_list = [Mock(request_gap=0.1), Mock(request_gap=0.2)]
        mock_request_info.request = mock_request
        mock_request_info.prefill_dp_max_seq_len = 1024
        self.router.request_map[REQUEST_KEY_PREFILL] = {}
        self.router.request_map[REQUEST_KEY_PREFILL][request_key] = mock_request_info
        
        # 模拟cloud_cut_instance
        mock_time_counter = Mock()
        mock_time_counter.get_cut_num.return_value = 3
        self.router.router_impl = Mock()
        self.router.router_impl.generator = Mock()
        self.router.router_impl.generator.model_wrapper = Mock()
        self.router.router_impl.generator.model_wrapper.model_runner = Mock()
        self.router.router_impl.generator.model_wrapper.model_runner.time_counter = mock_time_counter
        
        # 调用方法
        prefill_layers_divi_policy = self.router.update_prefill_layers_divi_num(request_key)
        
        # 验证结果
        expected_policy = [21, 21, 20]  # 62层切3份
        self.assertEqual(prefill_layers_divi_policy, expected_policy)
        self.assertEqual(mock_request_info.layers_divi_num, 3)
        mock_time_counter.get_cut_num.assert_called_once()

    def test_update_prefill_long_seq_data(self):
        # 测试update_prefill_long_seq_data函数
        request_key = 0
        prefill_dp_seq_len = 10000
        
        # 模拟request_map
        mock_request_info = Mock()
        mock_request_info.layers_divi_num = 4
        self.router.request_map[REQUEST_KEY_PREFILL] = {}
        self.router.request_map[REQUEST_KEY_PREFILL][request_key] = mock_request_info
        
        # 模拟prefill_chunk_instance
        self.router.prefill_chunk_instance = Mock()
        self.router.prefill_chunk_instance.get_chunk_len_policy.side_effect = [
            [2500, 2500, 2500, 2500],  # 非边缘设备的分块策略
            [2000, 2000, 2000, 2000, 2000]  # 边缘设备的分块策略
        ]
        
        # 模拟get_prefill_exec_metadata
        mock_metadata = Mock()
        self.router.get_prefill_exec_metadata = Mock(return_value=(mock_metadata, 1))
        
        # 模拟cloud_layer_num
        self.router.cloud_layer_num = 62
        
        # 清空prefill_metadata_queue
        self.router.prefill_metadata_queue = []
        
        # 调用方法
        self.router.update_prefill_long_seq_data(request_key, prefill_dp_seq_len)
        
        # 验证结果
        # 验证layers_divi_num被正确更新
        expected_layers_divi_num = round(4 / 4)  # 4层分割 / 4个分块
        self.assertEqual(mock_request_info.layers_divi_num, expected_layers_divi_num)
        
        # 验证get_chunk_len_policy被正确调用
        self.router.prefill_chunk_instance.get_chunk_len_policy.assert_any_call(prefill_dp_seq_len, False)
        self.router.prefill_chunk_instance.get_chunk_len_policy.assert_any_call(prefill_dp_seq_len, True)
        
        # 验证get_prefill_exec_metadata被调用的次数
        # 分块数 * 层分割数 = 4 * 1 = 4次
        self.assertEqual(self.router.get_prefill_exec_metadata.call_count, 4)
        
        # 验证prefill_metadata_queue被正确填充
        self.assertEqual(len(self.router.prefill_metadata_queue), 4)


if __name__ == "__main__":
    unittest.main()