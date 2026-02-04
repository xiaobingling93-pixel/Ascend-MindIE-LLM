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

import json
import queue
import sys
import time
from enum import IntEnum
from pathlib import Path

from mindie_llm.connector.common import send_model_execute_response
from mindie_llm.connector.common.input_metadata_builder import parse_all_dp_batches_seq_lens
from mindie_llm.connector.common.response_builder import ExecuteResponseBuilder
from mindie_llm.connector.common.model_execute_data_pb2 import ExecuteRequest, ExecuteType, ForwardType
from mindie_llm.connector.request_router.request_router import RequestRouter
from mindie_llm.utils.layerwise.communication import LwdCommunicationManager
from mindie_llm.utils.layerwise.share_memory import SharedMemoryManager
from mindie_llm.utils.layerwise.request_metadata import LwdMetadata, lwd_metadata_manager
from mindie_llm.utils.log.logging import logger
from mindie_llm.utils.prof.profiler import span_start, span_end

sys.path.append(str(Path(__file__).parent / "sync"))


MASTER_ID = 0   # 接收通信, 广播决策, 创建共享内存的主rank号
LONG_SEQ_LEN_MIN = 7500


class LastExecType(IntEnum):
    PREFILEE = 0
    DECODE = 1


class DecisionType(IntEnum):
    DO_NOTHING = 0
    DO_PREFILL_FIRST = 1
    DO_PREFILL = 2
    DO_PREFILL_LAST = 3
    DO_DECODE_FIRST = 4
    DO_DECODE = 5
    DO_DECODE_LAST = 6
    DO_CLEAN_UP = 7
    DO_CLEAN_EOS = 8
    WAIT_COMM = 9
    WAIT_DECODE = 10


class RequestRouterLwd(RequestRouter):
    def __init__(self):
        self.rank = None

        self.prefill_queue = queue.Queue()
        self.decode_queue = queue.Queue()
        self.clean_up_queue = queue.Queue() 
        self.clean_eos_queue = queue.Queue()

        self.decision_type = DecisionType.DO_NOTHING

        self.prefill_request = None
        self.prefill_shape = None
        self.decode_request = None
        self.decode_shape = None

        self.ctrl_comm = None
        self.data_comm = None

        self.mem_manager = None

        self.decode_comm_finish = False
        self.prefill_comm_finish = False
        self.prefill_comm_irecv_finish = False
        self.prefill_comm_tcp_finish_count = 0

        self.prefill_chunk_instance = None
        self.prefill_seq_len = 0
        self.prefill_batch_size = 0
        self.is_long_seq = False
        self.prefill_chunk_num = 1
        self.long_seq_start_idx = 0

        super().__init__()

    @staticmethod
    def get_lwd_models_config_dict(model_config):
        model_config_dict_json = model_config.get("models")
        models_config_dict = None
        if isinstance(model_config_dict_json, str):
            try:
                models_config_dict = json.loads(model_config_dict_json)
            except json.JSONDecodeError as e:
                message = "The 'models' field does not conform to JSON format. Please check."
                logger.warning(f'{message}, exception info: {e}')
                models_config_dict = None

        if models_config_dict is None:
            logger.info("[layerwiseDisaggregated] request router lwd initliaze error.")
            raise ValueError("request router init error, layerwiseDisaggregated models is None")

        return models_config_dict

    def initialize_diff(self, model_config, models_config_dict):
        pass

    def initialize(self, request_config):
        model_config = self.get_config_dict(request_config)
        config = self.get_model_impl_config(model_config)
        self.enable_dp_distributed = config.distributed_enable
        initialize_result = self.initialize_impl(config)

        edge_cloud_comm = LwdCommunicationManager()
        edge_cloud_comm.initialize(config, initialize_result, self.router_impl.generator)

        proto = ExecuteResponseBuilder.build_from_init_result(initialize_result)
        send_model_execute_response(proto)

        self.rank = config.rank
        self.ctrl_comm = edge_cloud_comm.ctrl_comm
        self.data_comm = edge_cloud_comm.data_comm
        self.prefill_chunk_instance = self.router_impl.generator.model_wrapper.model_runner.chunk_prefill_manager

        models_config_dict = self.get_lwd_models_config_dict(model_config)
        producer_glb_id_str = config.model_config.get('npu_device_ids', '0,1').split(',')[0]
        self.mem_manager = SharedMemoryManager(producer_glb_id_str)
        logger.info(f"[layerwiseDisaggregated] initliaze ok rank:{self.rank}, glb_rank:{producer_glb_id_str}")

        self.initialize_diff(model_config, models_config_dict)

    def curr_no_request(self):
        return self.prefill_queue.empty() and self.decode_queue.empty() and self.clean_up_queue.empty() and\
            self.prefill_request is None and self.decode_request is None

    def is_pd_inference_request(self, execute_type):
        return execute_type == ExecuteType.MODEL_INFER or execute_type == ExecuteType.MODEL_INFER_SECOND

    def save_pd_inference_request(self, execute_request: ExecuteRequest):
        forward_type = execute_request.execute_model_request.forward_type
        if forward_type == ForwardType.PREFILL:
            self.prefill_queue.put(execute_request)
        elif forward_type == ForwardType.DECODE:
            self.decode_queue.put(execute_request)
        else:
            self.router_impl.execute(execute_request)
        logger.info(f"[layerwiseDisaggregated] save pd request forward_type: {forward_type} rank: {self.rank}.")

    def do_other_request_now(self, execute_request: ExecuteRequest):
        execute_type = execute_request.execute_type
        if execute_type == ExecuteType.MODEL_INIT:
            self.initialize(execute_request.config)
            logger.info("[layerwiseDisaggregated][python thread: infer] model initialized.")
        elif execute_type == ExecuteType.MODEL_FINALIZE:
            self.router_impl.finalize()
            logger.info("[layerwiseDisaggregated][python thread: infer] model finalized.")
        else:
            logger.error(f"[layerwiseDisaggregated] Unknown execute_type {execute_type} now")
            raise RuntimeError("python doing an unknown execute inference type now")

    def save_inference_request(self, execute_request: ExecuteRequest):
        execute_type = execute_request.execute_type
        if self.is_pd_inference_request(execute_type):
            self.save_pd_inference_request(execute_request)
            return
        elif execute_type == ExecuteType.TEXT_GENERATOR_CLEANUP:
            self.clean_up_queue.put(execute_request)
        elif execute_type == ExecuteType.EOS_CLEANUP:
            self.clean_eos_queue.put(execute_request)
        else:
            self.do_other_request_now(execute_request)
        logger.info(f"[layerwiseDisaggregated] save other request execute_type: {execute_type} rank: {self.rank}.")

    def set_pd_curr_request(self):
        if self.prefill_request is None and not self.prefill_queue.empty():
            self.prefill_request = self.prefill_queue.get()
            self.prefill_seq_len = self.calc_seq_len(self.prefill_request)

        if self.decode_request is None and not self.decode_queue.empty():
            self.decode_request = self.decode_queue.get()

    def get_all_request(self):
        while not self.inference_queue.empty() or self.curr_no_request():
            if not self.clean_up_queue.empty():
                logger.info(f"[layerwiseDisaggregated] curr has clean up request, wait to clean up before "
                    f"get next request, rank{self.rank}.")
                break

            if self.inference_queue.empty():
                logger.info(f"[layerwiseDisaggregated] inference_queue is empty: {self.inference_queue.empty()}, "
                    f"rank:{self.rank}.")
            execute_request: ExecuteRequest = self.inference_queue.get()
            self.save_inference_request(execute_request)
        self.set_pd_curr_request()

    def recv_prefill(self):
        if self.prefill_comm_finish:
            return

        if self.prefill_comm_tcp_finish_count == 0: 
            self.ctrl_comm.recv_prefill() 
            self.prefill_comm_tcp_finish_count = self.ctrl_comm.prefill_comm_finish_tcp_count 

        self.prefill_comm_irecv_finish = True 
          
        if self.prefill_comm_tcp_finish_count > 0: 
            self.prefill_shape = self.ctrl_comm.parse_shape(self.ctrl_comm.prefill_recv_msg) 
            logger.info("[layerwiseDisaggregated] recv_prefill tcp comm finish.") 

        if self.prefill_comm_irecv_finish and self.prefill_comm_tcp_finish_count > 0: 
            self.prefill_comm_finish = True 
            self.prefill_comm_irecv_finish = False 
            self.prefill_comm_tcp_finish_count -= 1 
            self.ctrl_comm.prefill_comm_finish_tcp_count -= 1

    def recv_decode(self):
        self.ctrl_comm.recv_decode()
        self.decode_comm_finish = True

    def calc_decision_type(self):
        pass

    def arrange_exec_stage(self):
        metadata = LwdMetadata(0, 0, True, True, 1, 1, 62, False, 0, 0, 0, 0)
        lwd_metadata_manager.set_metadata(metadata)

    def decision_do_clean_up_type(self):
        has_clean_up = not self.clean_up_queue.empty()
        no_running_requests = self.prefill_request is None and self.decode_request is None
        no_queued_requests = self.prefill_queue.empty() and self.decode_queue.empty()
        if has_clean_up and no_running_requests and no_queued_requests:
            self.decision_type = DecisionType.DO_CLEAN_UP
            return True

        return False

    def broadcast_decision_type(self):
        if self.decision_type == DecisionType.DO_NOTHING:
            return

        self.mem_manager.write_list_memory([self.decision_type])

    def recv_decision_type(self):
        ctrl_tensor = self.mem_manager.read_list_memory(self.rank)
        logger.info(f"[layerwiseDisaggregated] recv_decision_type ctrl_tensor: {ctrl_tensor}, rank{self.rank}")
        if ctrl_tensor is None:
            self.decision_type = DecisionType.DO_NOTHING
            return

        self.decision_type = DecisionType(int(ctrl_tensor[0]))

    def exceute_inference_request(self):
        func = self.process_func.get(self.decision_type)
        if func:
            func()
        else:
            time.sleep(0.001)

    def do_inference(self):
        while True:
            prof = span_start("get_request")
            self.get_all_request() # 将inference_queue里的所有请求, 存到新增的三个队列中, 用于调度
            span_end(prof)

            if self.rank == MASTER_ID:
                self.recv_prefill() # 接收边侧发来的prefill tcp控制信号
                self.recv_decode() # 接收边侧发来的decode tcp控制信号

                # Determine the decision type for executing P/D based on the current information.
                self.calc_decision_type()
                logger.info(f"[layerwiseDisaggregated] decision_type:{self.decision_type}, "
                    f"has prefill:{self.prefill_request is not None}, prefill_comm_finish:{self.prefill_comm_finish}, "
                    f"has decode:{self.decode_request is not None}, decode_comm_finish:{self.decode_comm_finish}, "
                    f"clean_up_queue size:{self.clean_up_queue.qsize()}, "
                    f"clean_eos_queue size:{self.clean_eos_queue.qsize()}.")
                self.arrange_exec_stage()
                prof = span_start("broadcast_decision")
                self.broadcast_decision_type()
                span_end(prof)
            else:
                prof = span_start("recv_decision")
                self.recv_decision_type()
                span_end(prof)
                self.arrange_exec_stage()

            self.exceute_inference_request()

    def do_clean_up(self):
        while self.clean_up_queue.empty():
            self.get_all_request()

        execute_request: ExecuteRequest = self.clean_up_queue.get()
        self.router_impl.seq_ctrl(execute_request)
        logger.info(f"[layerwiseDisaggregated][python thread: infer] text generator clean up, rank{self.rank}.")

    def do_clean_eos(self):
        while self.clean_eos_queue.empty():
            self.get_all_request()

        execute_request: ExecuteRequest = self.clean_eos_queue.get()
        self.router_impl.seq_ctrl(execute_request)
        logger.info(f"[layerwiseDisaggregated][python thread: infer] text generator clean eos, rank{self.rank}.")

    def sum_nested(self, lst):
        total = 0
        for item in lst:
            if isinstance(item, list):
                total += self.sum_nested(item)
            else:
                total += item
        return total

    def calc_seq_len(self, execute_request: ExecuteRequest):
        seq_lens = parse_all_dp_batches_seq_lens(execute_request.execute_model_request.all_dp_batches_seq_lens)
        return self.sum_nested(seq_lens)

    def calc_batch_size(self, execute_request: ExecuteRequest):
        return len(execute_request.execute_model_request.seq_group_metadata_list)
