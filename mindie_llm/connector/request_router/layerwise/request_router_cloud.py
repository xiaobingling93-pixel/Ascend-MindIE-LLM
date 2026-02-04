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

import math
import os
import sys
import time
from enum import IntEnum
from pathlib import Path

from mindie_llm.utils.log.logging import logger
from mindie_llm.utils.prof.profiler import span_start, span_end
from mindie_llm.connector.common.model_execute_data_pb2 import ExecuteRequest, ExecuteType, ForwardType
from mindie_llm.utils.layerwise.request_metadata import LwdMetadata, lwd_metadata_manager
from mindie_llm.utils.layerwise.cloud_cut_inputdata import CloudCutInputData
from mindie_llm.connector.request_router.layerwise.request_router_lwd import RequestRouterLwd, LastExecType, \
    DecisionType, MASTER_ID, LONG_SEQ_LEN_MIN

sys.path.append(str(Path(__file__).parent / "sync"))

LAYERS_DIVI_MIN_NUM = 2


class CtrlTypePos(IntEnum):
    DECISION_TYPE = 0
    SHAPE_START = 1
    SHAPE_END = 2
    DIVI_NUM = 3
    CHUNK_NUM = 4
    SEQ_LEN = 5
    MAX_NUM = 6


class RequestRouterCloud(RequestRouterLwd):
    def __init__(self):
        self.last_execute_type = None
        self.before_last_execute_type = None

        self.prefill_exec_last_time = None
        self.prefill_exec_start_layer = 0
        self.prefill_exec_cnt = 0

        self.prefill_layers_divi_policy = [13] * 2 + [12] * 3
        self.prefill_layers_divi_num = int(os.getenv("PREFILL_CUT_NUM", "5"))
        self.prefill_layers_divi_switch = False if os.getenv("PREFILL_LAYERS_DIVI_SWITCH", "on") == "false" else True

        self.prefill_exec_chunk_cnt = 0
        self.prefill_chunk_policy = None

        self.total_layer_num = 64
        self.start_layer_num = 1
        self.end_layer_num = 1
        self.cloud_layer_num = 62

        self.isqwenvl = False   # 多模态

        self.process_func = {
            DecisionType.DO_PREFILL: self.do_prefill,
            DecisionType.DO_DECODE: self.do_decode,
            DecisionType.DO_CLEAN_UP: self.do_clean_up,
            DecisionType.DO_CLEAN_EOS: self.do_clean_eos,
        }

        super().__init__()

    def initialize_diff(self, model_config, models_config_dict):
        is_producer = True if self.rank == MASTER_ID else False
        card_num = models_config_dict.get('layerwiseDisaggregatedSlaveDeviceNum', 8)
        self.mem_manager.initialize(is_producer, card_num - 1)

        self.isqwenvl = True if model_config.get("model_name") == "qwen2_vl" else False

        self.start_layer_num = models_config_dict.get('startLayerNum', 1)
        self.end_layer_num = models_config_dict.get('endLayerNum', 1)
        model_runner_config = self.router_impl.generator.model_wrapper.model_runner.config
        self.total_layer_num = (64 if not hasattr(model_runner_config, 'num_hidden_layers')
            else model_runner_config.num_hidden_layers)
        self.cloud_layer_num = self.total_layer_num - self.start_layer_num - self.end_layer_num

        cloud_cut_instance = self.router_impl.generator.model_wrapper.model_runner.time_counter
        cloud_cut_instance.initialize("slave", self.rank, self.cloud_layer_num, LAYERS_DIVI_MIN_NUM)
        self.prepare_prefill_cut_policy(self.prefill_layers_divi_num)

        logger.info(f"[layerwiseDisaggregated] cloud initliaze ok rank:{self.rank}, is_producer:{is_producer}, "
            f"card_num:{card_num} start_layer_num:{self.start_layer_num}, end_layer_num:{self.end_layer_num}, "
            f"total_layer_num:{self.total_layer_num}, prefill_layers_divi_num = {self.prefill_layers_divi_num}, "
            f"prefill_layers_divi_policy = {self.prefill_layers_divi_policy}")

    def prepare_prefill_cut_policy(self, divi_num):
        self.prefill_layers_divi_num = divi_num
        divi_average_layer_num = int(self.cloud_layer_num / self.prefill_layers_divi_num)
        mod = self.cloud_layer_num % self.prefill_layers_divi_num
        self.prefill_layers_divi_policy = ([divi_average_layer_num] * self.prefill_layers_divi_num if mod == 0 else 
            [divi_average_layer_num + 1] * mod + [divi_average_layer_num] * (self.prefill_layers_divi_num - mod)
        )
    
    def prepare_prefill_chunk_policy(self, prefill_seq_len, prefill_chunk_num):
        average_prefill_chunk_seq_len = int(prefill_seq_len / prefill_chunk_num)
        mod = prefill_seq_len % prefill_chunk_num
        self.prefill_chunk_policy = ([average_prefill_chunk_seq_len] * prefill_chunk_num if mod == 0 else 
            [average_prefill_chunk_seq_len + 1] * mod + [average_prefill_chunk_seq_len] * (prefill_chunk_num - mod)
        )

    def get_prefill_gap_time_list(self, prefill_request):
        gap_time_list = []
        for seq_group_metadata in prefill_request.execute_model_request.seq_group_metadata_list:
            gap_time_list.append(seq_group_metadata.request_gap)
        return gap_time_list

    def update_prefill_layers_divi_num(self):
        if not self.prefill_layers_divi_switch:
            return

        cloud_cut_instance = self.router_impl.generator.model_wrapper.model_runner.time_counter
        gap_list = self.get_prefill_gap_time_list(self.prefill_request)
        self.prefill_layers_divi_num = cloud_cut_instance.get_cut_num(CloudCutInputData(self.prefill_seq_len, gap_list))
        self.prepare_prefill_cut_policy(self.prefill_layers_divi_num)

    def update_prefill_long_seq_data(self):
        if self.prefill_seq_len <= LONG_SEQ_LEN_MIN or self.prefill_batch_size > 1:
            return

        self.is_long_seq = True
        self.prefill_chunk_num = self.prefill_chunk_instance.map_prefill_chunk_num(self.prefill_seq_len)
        self.prepare_prefill_chunk_policy(self.prefill_seq_len, self.prefill_chunk_num)
        self.prefill_layers_divi_num = round(self.prefill_layers_divi_num / self.prefill_chunk_num)
        self.prepare_prefill_cut_policy(self.prefill_layers_divi_num)

    def set_pd_curr_request(self):
        if self.prefill_request is None and not self.prefill_queue.empty():
            self.prefill_request = self.prefill_queue.get()
            self.prefill_seq_len = self.calc_seq_len(self.prefill_request)
            self.prefill_batch_size = self.calc_batch_size(self.prefill_request)

            if self.rank == MASTER_ID:
                self.update_prefill_layers_divi_num()
                self.update_prefill_long_seq_data()

        if self.decode_request is None and not self.decode_queue.empty():
            if not self.clean_eos_queue.empty():    # 下一个D需要等clean_eos做完, 否则可能出现prefill before decode
                return
            self.decode_request = self.decode_queue.get()

    def decision_do_clean_eos_type(self):
        if not self.clean_eos_queue.empty() and self.decode_request is None:
            self.decision_type = DecisionType.DO_CLEAN_EOS
            return True

        return False

    def decision_do_prefill_or_decode_type(self):
        # If both d and p are present, interleave p/d execution based on the previous execution type; when both are
        # present by default, prioritize p.
        has_prefill_finish = self.prefill_request and self.prefill_comm_finish
        has_decode_finish = self.decode_request and self.decode_comm_finish
        if has_prefill_finish and has_decode_finish:
            if self.last_execute_type == LastExecType.DECODE:
                self.decision_type = DecisionType.DO_PREFILL
            elif self.last_execute_type == LastExecType.PREFILEE:
                self.decision_type = DecisionType.DO_DECODE
            else:
                self.decision_type = DecisionType.DO_PREFILL
            return True

        return False

    def decision_do_decode_type(self):
        has_decode_finish = self.decode_request and self.decode_comm_finish
        if has_decode_finish:
            self.decision_type = DecisionType.DO_DECODE
            return True

        return False

    def decision_do_prefill_type(self):
        # If only p is present, execute p: the first two p executions require a 10ms delay to allow decode an
        # opportunity for interleaving.
        #  p p p d p d p   ^  p d d d d d p d p d p   ^  p p p d p d
        #                 10ms                       10ms
        has_prefill_finish = self.prefill_request and self.prefill_comm_finish
        if has_prefill_finish:
            if self.last_execute_type == LastExecType.PREFILEE and \
                self.before_last_execute_type == LastExecType.DECODE:
                curr_time = time.time()
                if self.prefill_exec_last_time and \
                    curr_time - self.prefill_exec_last_time < 0.01:
                    self.decision_type = DecisionType.WAIT_DECODE
                else:
                    self.decision_type = DecisionType.DO_PREFILL
                return True

            self.decision_type = DecisionType.DO_PREFILL
            return True

        return False

    def calc_decision_type(self):
        if self.decision_do_clean_eos_type():
            return

        if self.decision_do_clean_up_type():
            return

        if self.decision_do_prefill_or_decode_type():
            return

        if self.decision_do_decode_type():
            return
        
        if self.decision_do_prefill_type():
            return

        self.decision_type = DecisionType.WAIT_COMM

    def broadcast_decision_type(self):
        if self.process_func.get(self.decision_type) is None:   # 无需广播的决策
            return

        ctrl_tensor = [0] * CtrlTypePos.MAX_NUM
        ctrl_tensor[CtrlTypePos.DECISION_TYPE] = self.decision_type
        ctrl_tensor[CtrlTypePos.SHAPE_START] = -1
        ctrl_tensor[CtrlTypePos.SHAPE_END] = -1
        ctrl_tensor[CtrlTypePos.DIVI_NUM] = self.prefill_layers_divi_num
        ctrl_tensor[CtrlTypePos.CHUNK_NUM] = self.prefill_chunk_num
        ctrl_tensor[CtrlTypePos.SEQ_LEN] = self.prefill_seq_len
        self.mem_manager.write_list_memory(ctrl_tensor)

    def recv_do_prefill_type_update_policy(self, ctrl_tensor):
        self.prefill_layers_divi_num = int(ctrl_tensor[CtrlTypePos.DIVI_NUM])
        self.prepare_prefill_cut_policy(self.prefill_layers_divi_num)
        self.prefill_chunk_num = int(ctrl_tensor[CtrlTypePos.CHUNK_NUM])
        self.prefill_seq_len = int(ctrl_tensor[CtrlTypePos.SEQ_LEN])
        if self.prefill_chunk_num > 1:
            self.is_long_seq = True
            self.prepare_prefill_chunk_policy(self.prefill_seq_len, self.prefill_chunk_num)

    def recv_decision_type(self):
        ctrl_tensor = self.mem_manager.read_list_memory(self.rank)
        logger.info(f"[layerwiseDisaggregated] recv decision type, ctrl_tensor:{ctrl_tensor}, rank{self.rank}")
        if ctrl_tensor is None:
            self.decision_type = DecisionType.DO_NOTHING
            return

        self.decision_type = DecisionType(ctrl_tensor[CtrlTypePos.DECISION_TYPE])
        shape = ctrl_tensor[CtrlTypePos.SHAPE_START: CtrlTypePos.SHAPE_END + 1]
        if self.decision_type == DecisionType.DO_DECODE:
            self.ctrl_comm.decode_recv_msg = self.ctrl_comm.shape_to_msg(shape)
        elif self.decision_type == DecisionType.DO_PREFILL:
            if self.prefill_exec_cnt == 0:
                self.recv_do_prefill_type_update_policy(ctrl_tensor)
            self.ctrl_comm.prefill_recv_msg = self.ctrl_comm.shape_to_msg(shape)
        return

    def do_prefill_end_clear_data(self):
        if self.prefill_exec_cnt >= self.prefill_layers_divi_num and \
            (not self.is_long_seq or self.prefill_exec_chunk_cnt >= self.prefill_chunk_num):
            self.prefill_exec_cnt = 0
            self.prefill_exec_start_layer = 0
            self.prefill_request = None
            self.prefill_comm_finish = False
            self.is_long_seq = False
            self.prefill_exec_chunk_cnt = 0
            self.prefill_chunk_num = 0
            self.long_seq_start_idx = 0

    def do_prefill(self):
        prof = span_start("Prefill")
        logger.info(f"[layerwiseDisaggregated] execute do_prefill before, rank:{self.rank}.")
        while self.prefill_request is None:
            self.get_all_request()
        prefill_start_time = time.time()
        self.router_impl.execute(self.prefill_request)
        prefill_end_time = time.time()
        logger.info(f"[layerwiseDisaggregated] cloud do_prefill exec layer cnt:{self.prefill_exec_cnt}, "
            f"exec chunk cnt:{self.prefill_exec_chunk_cnt} prefill_chunk_num:{self.prefill_chunk_num}"
            f"time exec cost {1000 * (prefill_end_time - prefill_start_time)}ms, rank{self.rank}.")
        self.do_prefill_end_clear_data()
        self.before_last_execute_type = self.last_execute_type
        self.last_execute_type = LastExecType.PREFILEE
        self.prefill_exec_last_time = prefill_end_time
        span_end(prof)
        return

    def do_decode(self):
        prof = span_start("Decode")
        logger.info(f"[layerwiseDisaggregated] execute do_decode before, rank:{self.rank}.")
        while self.decode_request is None:
            self.get_all_request()
        decode_start_time = time.time()
        self.router_impl.execute(self.decode_request)
        decode_end_time = time.time()
        self.before_last_execute_type = self.last_execute_type
        self.last_execute_type = LastExecType.DECODE
        self.decode_comm_finish = False 
        self.ctrl_comm.decode_comm_finish = False
        self.decode_request = None
        logger.info(f"[layerwiseDisaggregated] cloud do_decode, time exec cost "
            f"{1000 * (decode_end_time - decode_start_time)}ms, rank{self.rank}.")
        span_end(prof)
        return

    def get_prefill_exec_metadata(self):
        chunk_end_input_offset = 0
        chunk_next_end_input_offset = 0
        if self.is_long_seq:
            chunk_end_input_offset = self.long_seq_start_idx + self.prefill_chunk_policy[self.prefill_exec_chunk_cnt]
            chunk_next_end_input_offset = (
                chunk_end_input_offset + self.prefill_chunk_policy[self.prefill_exec_chunk_cnt + 1] 
                if chunk_end_input_offset < self.prefill_seq_len else 0
            )

        exec_end_layer = self.prefill_exec_start_layer + self.prefill_layers_divi_policy[self.prefill_exec_cnt]
        metadata = LwdMetadata(self.prefill_exec_start_layer, exec_end_layer, False, True, 1, 1,
            self.cloud_layer_num, self.is_long_seq, self.long_seq_start_idx,
            chunk_end_input_offset, chunk_next_end_input_offset, self.prefill_seq_len)
        
        self.prefill_exec_start_layer = exec_end_layer
        self.prefill_exec_cnt += 1

        if exec_end_layer >= self.cloud_layer_num and self.is_long_seq and \
            chunk_end_input_offset < self.prefill_seq_len:
            self.prefill_exec_start_layer = 0
            self.prefill_exec_cnt = 0
            self.long_seq_start_idx = chunk_end_input_offset
            self.prefill_exec_chunk_cnt += 1
        # The final chunk of prefill requires a dummy return.
        elif exec_end_layer >= self.cloud_layer_num and \
            (not self.is_long_seq or chunk_end_input_offset >= self.prefill_seq_len):
            self.prefill_exec_start_layer = 0
            self.long_seq_start_idx = 0
            self.prefill_exec_chunk_cnt += 1
            metadata.end_of_generate_token = True

        return metadata

    def arrange_exec_stage(self):
        if self.decision_type == DecisionType.DO_DECODE:
            metadata = LwdMetadata(0, self.cloud_layer_num, True, False, 1, 1, self.cloud_layer_num, False, 0, 0, 0, 0)
            lwd_metadata_manager.set_metadata(metadata)
            logger.info(f"[layerwiseDisaggregated]exec decode, rank{self.rank} set metadata: {metadata}")
        elif self.decision_type == DecisionType.DO_PREFILL:
            metadata = self.get_prefill_exec_metadata()
            lwd_metadata_manager.set_metadata(metadata)
            logger.info(f"[layerwiseDisaggregated]exec prefill, rank{self.rank} set metadata: {metadata}")

    def accept_prefill_prepare(self, execute_request: ExecuteRequest):
        seq_len = self.calc_seq_len(execute_request)
        batch_size = self.calc_batch_size(execute_request)
        if self.isqwenvl:
            seq_len = seq_len * 2
        prefill_chunk_num = 1
        if seq_len > LONG_SEQ_LEN_MIN and batch_size == 1:
            prefill_chunk_num = self.prefill_chunk_instance.map_prefill_chunk_num(seq_len)
        self.data_comm.prefill_seq_len_queue.put(math.ceil(seq_len / prefill_chunk_num))

        if not self.data_comm.need_set_prefill_device:
            self.data_comm.need_set_prefill_device = True

        self.data_comm.p_shape[self.data_comm.recv_index] = self.data_comm.prefill_seq_len_queue.get()
        self.data_comm.recv_hidden('p', self.data_comm.p_shape)

        logger.info(f"[layerwiseDisaggregated] cloud prefill seq len putted, rank{self.rank} seq_len: {seq_len}")

    def accept_decode_prepare(self, execute_request: ExecuteRequest):
        batch_size = self.calc_batch_size(execute_request)
        if self.isqwenvl:
            batch_size = batch_size + 1
        self.data_comm.decode_batch_size_queue.put(batch_size)

        if not self.data_comm.need_set_decode_device:
            self.data_comm.need_set_decode_device = True

        with self.data_comm.lock:
            logger.info(f"[layerwiseDisaggregated] req_router, pre recv "
                f"{self.data_comm.decode_batch_size_queue.qsize()} {self.data_comm.flag_pre_recv}")
            if self.data_comm.flag_pre_recv:
                self.data_comm.d_shape = self.data_comm.decode_batch_size_queue.get()
                self.data_comm.recv_hidden('d', self.data_comm.d_shape)
                self.data_comm.flag_pre_recv = False

        logger.info(f"[layerwiseDisaggregated] cloud decode batch size putted, "
            f"rank{self.rank} batch_size: {batch_size}")

    def accept(self, execute_request: ExecuteRequest):
        if execute_request.execute_type == ExecuteType.MODEL_INFER:
            forward_type = execute_request.execute_model_request.forward_type
            if forward_type == ForwardType.PREFILL:
                self.accept_prefill_prepare(execute_request)
            elif forward_type == ForwardType.DECODE:
                self.accept_decode_prepare(execute_request)
            self.inference_queue.put(execute_request)
        elif execute_request.execute_type == ExecuteType.PD_LINK:
            self.link_queue.put(execute_request)
        elif execute_request.execute_type == ExecuteType.KV_TRANSFER:
            self.transfer_queue.put(execute_request)
        elif execute_request.execute_type == ExecuteType.LORA_OPERATION:
            self.command_queue.put(execute_request)
        else:
            self.inference_queue.put(execute_request)
