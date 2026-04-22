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

import sys
import time
from itertools import accumulate
from enum import IntEnum
from pathlib import Path

from mindie_llm.utils.log.logging import logger
from mindie_llm.utils.prof.profiler import span_start, span_end
from mindie_llm.connector.common.model_execute_data_pb2 import ForwardType
from mindie_llm.utils.layerwise.request_metadata import LwdMetadata, lwd_metadata_manager
from mindie_llm.connector.request_router.layerwise.request_router_lwd import (
    DecisionMetadata,
    RequestRouterLwd,
    DecisionType,
    MASTER_ID,
    RequestInfo,
    REQUEST_KEY_PREFILL,
    REQUEST_KEY_DECODE,
)

sys.path.append(str(Path(__file__).parent / "sync"))


class CtrlTypePos(IntEnum):
    DECISION_TYPE = 0
    SHAPE_START = 1
    SHAPE_END = 2
    REQUEST_KEY = 3
    MAX_NUM = 4


class RequestRouterEdge(RequestRouterLwd):
    def __init__(self, parent_pid):
        """
        Wait for the next Decode with timing;
        non-empty indicates the previous execution was a D-end task.
        """
        self.force_wait_d_time = None

        self.process_func = {
            DecisionType.DO_PREFILL_FIRST: self.do_prefill_first,
            DecisionType.DO_PREFILL_LAST: self.do_prefill_last,
            DecisionType.DO_DECODE_FIRST: self.do_decode_first,
            DecisionType.DO_DECODE_LAST: self.do_decode_last,
            DecisionType.DO_CLEAN_UP: self.do_clean_up,
            DecisionType.DO_CLEAN_EOS: self.do_clean_eos,
        }

        super().__init__(parent_pid)

    def do_prefill_first(self, request_key):
        prof = span_start("Prefill_first")
        logger.info(f"[layerwiseDisaggregated] execute prefill_first before, rank:{self.rank}.")
        metadata = lwd_metadata_manager.get_metadata()
        request = self.request_map[REQUEST_KEY_PREFILL][request_key].request
        if metadata.is_dummy_batch:  # 如果取出来的是结束了的, 就跑一个空batch, 否则执行prefill
            request.execute_model_request.forward_type = ForwardType.DUMMY
            self.router_impl.execute(request)
            request.execute_model_request.forward_type = ForwardType.PREFILL
            logger.info(f"[layerwiseDisaggregated] execute prefill_first dummy end, rank:{self.rank}.")
        else:
            self.router_impl.execute(request)
            logger.info(f"[layerwiseDisaggregated] execute prefill_first end, rank:{self.rank}.")
        span_end(prof)

    def do_prefill_last(self, request_key):
        prof = span_start("Prefill_last")
        logger.info(f"[layerwiseDisaggregated] execute prefill_last before, rank:{self.rank}.")
        metadata = lwd_metadata_manager.get_metadata()
        request = self.request_map[REQUEST_KEY_PREFILL][request_key].request
        if metadata.is_dummy_batch:  # 如果取出来的是结束了的, 就跑一个空batch, 否则执行prefill
            request.execute_model_request.forward_type = ForwardType.DUMMY
            self.router_impl.execute(request)
            request.execute_model_request.forward_type = ForwardType.PREFILL
            logger.info(f"[layerwiseDisaggregated] execute prefill_last dummy end, rank:{self.rank}.")
        else:
            self.router_impl.execute(request)
            logger.info(f"[layerwiseDisaggregated] execute prefill_last end, rank:{self.rank}.")

        self.prefill_comm_finish = False
        if metadata.end_of_generate_token:
            del self.request_map[REQUEST_KEY_PREFILL][request_key]
        span_end(prof)

    def do_decode_first(self, request_key):
        request = self.request_map[REQUEST_KEY_DECODE][request_key].request
        prof = span_start("Decode_first")
        logger.info(f"[layerwiseDisaggregated] execute decode_first before, rank:{self.rank}.")
        self.router_impl.execute(request)
        logger.info(f"[layerwiseDisaggregated] execute decode_first end, rank:{self.rank}.")
        span_end(prof)

    def do_decode_last(self, request_key):
        prof = span_start("Decode_last")
        logger.info(f"[layerwiseDisaggregated] execute decode_last before, rank:{self.rank}.")
        request = self.request_map[REQUEST_KEY_DECODE][request_key].request
        metadata = lwd_metadata_manager.get_metadata()
        self.router_impl.execute(request)
        if self.rank == MASTER_ID:
            self.force_wait_d_time = time.time()  # Wait for the next Decode with timing.
        logger.info(f"[layerwiseDisaggregated] execute decode_last end, rank:{self.rank}.")
        self.decode_comm_finish = False
        self.ctrl_comm.decode_comm_finish = False
        if metadata.end_of_generate_token:
            del self.request_map[REQUEST_KEY_DECODE][request_key]
        span_end(prof)

    def do_clean_eos(self, request_key=None):
        while self.clean_eos_queue.empty():
            time.sleep(0.001)
            self.get_all_request()

        self.clean_eos_queue.get()  # 边侧推出eos会自动清理cache, 下eos请求下给云侧清理cache使用
        logger.info(f"[layerwiseDisaggregated][python thread: infer] text generator clean eos, rank{self.rank}.")

    def recv_prefill(self):
        if self.prefill_comm_finish:
            return

        # 边侧一次recv_prefill, 可能会有多个chunk完成(长序列场景)
        if self.prefill_comm_tcp_finish_count > 0:
            self.prefill_comm_finish = True
            self.prefill_comm_tcp_finish_count -= 1
            return

        self.ctrl_comm.recv_prefill()
        self.prefill_comm_tcp_finish_count = self.ctrl_comm.prefill_comm_finish_tcp_count
        if self.prefill_comm_tcp_finish_count <= 0:
            return

        # 边侧一次recv_prefill, 可以完成几次prefill last的处理; chunk_index_size默认为1, 兼容长短序列
        chunk_index_size = self.prefill_metadata_queue[0][2].chunk_index_size if self.prefill_metadata_queue else 1
        self.prefill_comm_tcp_finish_count *= chunk_index_size

        # 清除接收完成的次数
        self.ctrl_comm.prefill_comm_finish_tcp_count = 0

        logger.info(
            f"[layerwiseDisaggregated] recv_prefill tcp comm finish, chunk_index_size:{chunk_index_size} "
            f"prefill_comm_tcp_finish_count:{self.prefill_comm_tcp_finish_count} rank:{self.rank}"
        )
        # 第一次使用, 减1, 后续使用, 在上面减1
        self.prefill_comm_tcp_finish_count -= 1
        self.prefill_comm_finish = True

    def recv_decode(self):
        if self.decode_comm_finish:
            return
        self.ctrl_comm.recv_decode()
        self.decode_comm_finish = self.ctrl_comm.decode_comm_finish
        logger.info(f"[layerwiseDisaggregated] decode_comm_finish = {self.decode_comm_finish} rank:{self.rank}")

    def recv_ctrl_msg(self):
        # 这里应该是多DP的主节点需要接收, 其他卡无需接收
        is_need_recv = self.lwd_multi_nodes_enable or (not self.lwd_multi_nodes_enable and self.rank == MASTER_ID)

        if is_need_recv:
            self.recv_prefill()  # 接收对方发来的prefill tcp控制信号
            self.recv_decode()  # 接收对方发来的decode tcp控制信号

    def check_10ms_for_next_decode(self):
        if self.force_wait_d_time is None:
            return
        wait_d_time_gap = time.time() - self.force_wait_d_time
        if wait_d_time_gap > 0.01:
            logger.info(
                f"[layerwiseDisaggregated] Force wait decode state exit, wait time: {wait_d_time_gap * 1000} ms."
            )
            self.force_wait_d_time = None

    def calc_prefill_priority_decision_type(self, prefill_metadata=None, decode_metadata=None):
        decision_type = DecisionType.WAIT_COMM

        if self.force_wait_d_time and decode_metadata and decode_metadata.start_exec_layer == 0:
            decision_type = DecisionType.DO_DECODE_FIRST
            self.force_wait_d_time = None
            logger.info("[layerwiseDisaggregated] Force to do decode first, wait time over 10 ms.")
        elif self.force_wait_d_time:
            decision_type = DecisionType.WAIT_DECODE
        elif prefill_metadata and prefill_metadata.start_exec_layer == 0:
            decision_type = DecisionType.DO_PREFILL_FIRST
        elif prefill_metadata and prefill_metadata.end_exec_layer == 1 and self.prefill_comm_finish:
            decision_type = DecisionType.DO_PREFILL_LAST
        elif decode_metadata and decode_metadata.start_exec_layer == 0:
            decision_type = DecisionType.DO_DECODE_FIRST
        elif decode_metadata and decode_metadata.end_exec_layer == 1 and self.decode_comm_finish:
            decision_type = DecisionType.DO_DECODE_LAST

        return decision_type

    def calc_decode_priority_decision_type(self, prefill_metadata=None, decode_metadata=None):
        decision_type = DecisionType.WAIT_COMM

        if decode_metadata and decode_metadata.start_exec_layer == 0:
            decision_type = DecisionType.DO_DECODE_FIRST
        elif decode_metadata and decode_metadata.end_exec_layer == 1 and self.decode_comm_finish:
            decision_type = DecisionType.DO_DECODE_LAST
        elif prefill_metadata and prefill_metadata.start_exec_layer == 0:
            decision_type = DecisionType.DO_PREFILL_FIRST
        elif prefill_metadata and prefill_metadata.end_exec_layer == 1 and self.prefill_comm_finish:
            decision_type = DecisionType.DO_PREFILL_LAST

        return decision_type

    def decision_do_clean_eos_type(self):
        decision_type = DecisionType.WAIT_COMM
        if not self.clean_eos_queue.empty():
            decision_type = DecisionType.DO_CLEAN_EOS
            return decision_type

        return decision_type

    def calc_decision_type(self):
        self.check_10ms_for_next_decode()
        decision_type = DecisionType.WAIT_COMM

        if self.decision_do_clean_eos_type() == DecisionType.DO_CLEAN_EOS:
            decision_type = DecisionType.DO_CLEAN_EOS
            return decision_type

        if self.decision_do_clean_up_type() == DecisionType.DO_CLEAN_UP:
            decision_type = DecisionType.DO_CLEAN_UP
            return decision_type
        """
        If the previous task was a decode-last and a new decode-first task arrives within 10ms,
        execute the decode-first task.
        """
        prefill_metadata = self.prefill_metadata_queue[0][1] if self.prefill_metadata_queue else None
        decode_metadata = self.decode_metadata_queue[0][1] if self.decode_metadata_queue else None
        if prefill_metadata and prefill_metadata.is_long_seq:
            decision_type = self.calc_decode_priority_decision_type(prefill_metadata, decode_metadata)
        else:
            decision_type = self.calc_prefill_priority_decision_type(prefill_metadata, decode_metadata)

        return decision_type

    def get_cloud_eq_edge_prefill_chunk_num_list(self, prefill_chunk_policy, cloud_chunk_policy):
        # 找到云侧一段chunk等于边侧多少段chunk数的list
        eq_chunk_num_list = []

        cloud_index = 1
        last_chunk_end_index = 0
        for cloud_end_offset in cloud_chunk_policy[cloud_index:]:
            for i in range(last_chunk_end_index, len(prefill_chunk_policy)):
                end_offset = prefill_chunk_policy[i]
                if end_offset >= cloud_end_offset:
                    eq_chunk_num_list.append(i - last_chunk_end_index)
                    last_chunk_end_index = i
                    break

        return eq_chunk_num_list

    # 小于chunk数的超短序列, 除了第一个数据为真以外, 其他数据全部是dummy_batch
    def prepare_lt_chunk_num_prefill_metadata(self, curr_dp_seq_len, chunk_num, is_p_first, request_key):
        metadata_list = []
        start_or_end_layer = 0 if is_p_first else 1
        end_of_generate_token = not is_p_first

        metadata = LwdMetadata(
            request_key,
            start_or_end_layer,
            start_or_end_layer,
            end_of_generate_token,
            True,
            False,
            True,
            0,
            True,
            0,
            curr_dp_seq_len,
            0,
            curr_dp_seq_len,
            False,
        )
        metadata_list.append(metadata)

        for i in range(1, chunk_num):
            is_last_chunk = True if i == chunk_num - 1 else False
            metadata = LwdMetadata(
                request_key,
                start_or_end_layer,
                start_or_end_layer,
                False,
                True,
                True,
                True,
                0,
                True,
                curr_dp_seq_len,
                curr_dp_seq_len,
                0,
                curr_dp_seq_len,
                is_last_chunk,
            )
            metadata_list.append(metadata)

    def prepare_chunk_prefill_metadata(self, prefill_chunk_policy, curr_dp_seq_len, dp_empty, is_p_first, request_key):
        chunk_num = len(prefill_chunk_policy) - 1
        # 如果当前dp域是空, 或者当前dp域的seq len小于等于短序列最小值的一半, 则只生成第一份有用的metadata
        if dp_empty or curr_dp_seq_len <= self.get_long_seq_len_min() / 2:
            return self.prepare_lt_chunk_num_prefill_metadata(curr_dp_seq_len, chunk_num, is_p_first, request_key)

        metadata_list = []
        start_or_end_layer = 0 if is_p_first else 1
        for i in range(chunk_num):
            start_offset = prefill_chunk_policy[i]
            end_offset = prefill_chunk_policy[i + 1]

            is_last_chunk = True if i == chunk_num - 1 else False
            end_of_generate_token = True if is_last_chunk and not is_p_first else False

            metadata = LwdMetadata(
                request_key,
                start_or_end_layer,
                start_or_end_layer,
                end_of_generate_token,
                True,
                False,
                False,
                0,
                True,
                start_offset,
                end_offset,
                0,
                curr_dp_seq_len,
                is_last_chunk,
            )
            metadata_list.append(metadata)
        return metadata_list

    def generate_send_metadata_to_queue(self, request_key, **kwargs):
        cloud_eq_edge_chunk_num_list = kwargs["cloud_eq_edge_chunk_num_list"]
        p_first_metadata_list = kwargs["p_first_metadata_list"]
        p_last_metadata_list = kwargs["p_last_metadata_list"]
        last_chunk_num_eq_cloud = kwargs["last_chunk_num_eq_cloud"]
        cloud_chunk_len_policy = kwargs["cloud_chunk_len_policy"]

        tmp_metadata_list = []
        if last_chunk_num_eq_cloud > 0 and len(self.prefill_metadata_queue) >= last_chunk_num_eq_cloud:
            while last_chunk_num_eq_cloud > 0:
                last_chunk_num_eq_cloud -= 1
                tmp_metadata_list.append(self.prefill_metadata_queue.pop())

        chunk_group_size = len(cloud_eq_edge_chunk_num_list)

        # 放入第一段首
        first_chunk_num = cloud_eq_edge_chunk_num_list[0]
        for i in range(first_chunk_num):
            long_seq_metadata = DecisionMetadata(chunk_group_size, 0, first_chunk_num, i)
            metadata = p_first_metadata_list.pop(0)
            self.prefill_metadata_queue.append((request_key, metadata, long_seq_metadata))

        # 如果之前有N个尾, 重新放入N个尾
        while len(tmp_metadata_list) > 0:
            tmp_metadata = tmp_metadata_list.pop()
            self.prefill_metadata_queue.append(tmp_metadata)

        # 交叉放入首和尾
        for i in range(1, chunk_group_size):
            first_chunk_num = cloud_eq_edge_chunk_num_list[i]
            last_chunk_num = cloud_eq_edge_chunk_num_list[i - 1]

            for j in range(first_chunk_num):
                long_seq_metadata = DecisionMetadata(chunk_group_size, i, first_chunk_num, j)
                metadata = p_first_metadata_list.pop(0)
                self.prefill_metadata_queue.append((request_key, metadata, long_seq_metadata))
            # 接收下一段请求
            recv_list = [(0, cloud_chunk_len_policy[i - 1])]
            self.prefill_metadata_queue[-1][1].long_seq_recv_list = recv_list

            hidden_start_pos = 0
            for j in range(last_chunk_num):
                long_seq_metadata = DecisionMetadata(chunk_group_size, i - 1, last_chunk_num, j)
                metadata = p_last_metadata_list.pop(0)
                # 保存从通信中需要获取的起点
                metadata.hidden_start_pos = hidden_start_pos
                self.prefill_metadata_queue.append((request_key, metadata, long_seq_metadata))
                # 求出起止点
                hidden_start_pos += metadata.long_seq_end_idx - metadata.long_seq_start_idx

        # 接收最后一段P尾
        recv_list = [(0, cloud_chunk_len_policy[-1])]
        self.prefill_metadata_queue[-1][1].long_seq_recv_list = recv_list
        # 放入最后一段尾
        last_chunk_num = cloud_eq_edge_chunk_num_list[-1]
        hidden_start_pos = 0
        for j in range(last_chunk_num):
            long_seq_metadata = DecisionMetadata(chunk_group_size, chunk_group_size - 1, last_chunk_num, j)
            metadata = p_last_metadata_list.pop(0)
            metadata.hidden_start_pos = hidden_start_pos
            self.prefill_metadata_queue.append((request_key, metadata, long_seq_metadata))
            hidden_start_pos += metadata.long_seq_end_idx - metadata.long_seq_start_idx

    def prepare_chunk_prefill_metadata_queue(self, curr_dp_seq_len, dp_empty, request_key):
        chunk_len_policy = self.prefill_chunk_instance.get_chunk_len_policy(curr_dp_seq_len, True)
        prefill_chunk_policy = [0] + list(accumulate(chunk_len_policy))
        cloud_chunk_len_policy = self.prefill_chunk_instance.get_chunk_len_policy(curr_dp_seq_len, False)
        cloud_chunk_policy = [0] + list(accumulate(cloud_chunk_len_policy))

        cloud_eq_edge_chunk_num_list = self.get_cloud_eq_edge_prefill_chunk_num_list(
            prefill_chunk_policy, cloud_chunk_policy
        )
        p_first_metadata_list = self.prepare_chunk_prefill_metadata(
            prefill_chunk_policy, curr_dp_seq_len, dp_empty, True, request_key
        )
        p_last_metadata_list = self.prepare_chunk_prefill_metadata(
            prefill_chunk_policy, curr_dp_seq_len, dp_empty, False, request_key
        )

        # 上一个请求最后云一大段等于边几个小段长
        edge_eq_last_chunk_num = (
            self.prefill_metadata_queue[-1][2].chunk_index_size if self.prefill_metadata_queue else 0
        )
        # 生成所有metadata并排序
        self.generate_send_metadata_to_queue(
            request_key,
            cloud_eq_edge_chunk_num_list=cloud_eq_edge_chunk_num_list,
            p_first_metadata_list=p_first_metadata_list,
            p_last_metadata_list=p_last_metadata_list,
            last_chunk_num_eq_cloud=edge_eq_last_chunk_num,
            cloud_chunk_len_policy=cloud_chunk_len_policy,
        )

        logger.info(
            f"[layerwiseDisaggregated] edge_eq_last_chunk_num: {edge_eq_last_chunk_num} curr_dp_seq_len: "
            f"{curr_dp_seq_len} prefill_chunk_policy: {prefill_chunk_policy} "
            f"chunk_len_policy: {chunk_len_policy} "
            f"cloud_chunk_len_policy: {cloud_chunk_len_policy} cloud_chunk_policy:{cloud_chunk_policy} "
            f"cloud_eq_edge_chunk_num_list:{cloud_eq_edge_chunk_num_list} rank {self.rank}"
        )

    def arrange_exec_stage(self, decision_type: DecisionType):
        metadata = None
        request_key = -1
        if decision_type == DecisionType.DO_DECODE_FIRST or decision_type == DecisionType.DO_DECODE_LAST:
            request_key, metadata = self.decode_metadata_queue.popleft()
        elif decision_type == DecisionType.DO_PREFILL_FIRST or decision_type == DecisionType.DO_PREFILL_LAST:
            request_key, metadata, _ = self.prefill_metadata_queue.popleft()
        if metadata:
            lwd_metadata_manager.set_metadata(metadata)
            logger.info(
                f"[layerwiseDisaggregated] arrange exec stage decision_type:{decision_type.name} \
                metadata{metadata}, rank{self.rank}"
            )
        return request_key

    def broadcast_decision_type(self, decision_type: DecisionType, request_key):
        if self.process_func.get(decision_type) is None:  # 无需广播的决策
            return

        ctrl_tensor = [0] * CtrlTypePos.MAX_NUM
        ctrl_tensor[CtrlTypePos.DECISION_TYPE] = decision_type
        ctrl_tensor[CtrlTypePos.SHAPE_START] = -1
        ctrl_tensor[CtrlTypePos.SHAPE_END] = -1
        ctrl_tensor[CtrlTypePos.REQUEST_KEY] = request_key
        self.mem_manager.write_list_memory(ctrl_tensor)

    def recv_decision_type(self):
        ctrl_tensor = self.mem_manager.read_list_memory(self.rank)
        logger.info(f"[layerwiseDisaggregated] recv_decision_type ctrl_tensor {ctrl_tensor}, rank{self.rank}")
        if ctrl_tensor is None:
            decision_type = DecisionType.WAIT_COMM
            return decision_type

        decision_type = DecisionType(int(ctrl_tensor[CtrlTypePos.DECISION_TYPE]))
        request_key = int(ctrl_tensor[CtrlTypePos.REQUEST_KEY])
        while (
            decision_type == DecisionType.DO_PREFILL_FIRST and request_key not in self.request_map[REQUEST_KEY_PREFILL]
        ) or (
            decision_type == DecisionType.DO_DECODE_FIRST and request_key not in self.request_map[REQUEST_KEY_DECODE]
        ):
            self.get_all_request()

        if decision_type == DecisionType.DO_PREFILL_FIRST:
            if not self.request_map[REQUEST_KEY_PREFILL][request_key].prefill_metadata_ready:
                self.prepare_prefill_request(request_key)
                self.request_map[REQUEST_KEY_PREFILL][request_key].prefill_metadata_ready = True

        shape = ctrl_tensor[CtrlTypePos.SHAPE_START : CtrlTypePos.SHAPE_END + 1]
        if decision_type == DecisionType.DO_DECODE_LAST:
            self.ctrl_comm.decode_recv_msg = self.ctrl_comm.shape_to_msg(shape)
        elif decision_type == DecisionType.DO_PREFILL_LAST:
            self.ctrl_comm.prefill_recv_msg = self.ctrl_comm.shape_to_msg(shape)
        return decision_type

    def prepare_prefill_request(self, request_key):
        prefill_request = self.request_map[REQUEST_KEY_PREFILL][request_key].request
        prefill_request_info = RequestInfo(request=prefill_request)

        prefill_dp_seq_len = self.calc_curr_dp_seq_len(prefill_request)
        prefill_dp_max_seq_len = self.calc_max_seq_len(prefill_request)
        if prefill_dp_seq_len == 0:  # 当前batch是空
            prefill_dp_seq_len = 1  # 长度至少为1, 构造陪跑时的长度
            prefill_request_info.dp_empty = True
        if self.cp_size > 1:  # cp与dp实现方式基本是一样的
            prefill_dp_seq_len = self.calc_cp_seq_len(prefill_request)

        prefill_request_info.prefill_dp_len = prefill_dp_seq_len
        prefill_request_info.prefill_dp_max_seq_len = prefill_dp_max_seq_len
        self.request_map[REQUEST_KEY_PREFILL][request_key] = prefill_request_info

        if prefill_dp_max_seq_len > self.get_long_seq_len_min():
            self.prepare_chunk_prefill_metadata_queue(prefill_dp_seq_len, prefill_request_info.dp_empty, request_key)
        else:
            prefill_first_metadata = LwdMetadata(
                request_key, 0, 0, False, True, False, False, 0, False, 0, 0, 0, 0, False
            )
            prefill_last_metadata = LwdMetadata(
                request_key, 1, 1, True, True, False, False, 0, False, 0, 0, 0, 0, False
            )
            if self.prefill_metadata_queue:
                last_metadata = self.prefill_metadata_queue.pop()
                self.prefill_metadata_queue.append((request_key, prefill_first_metadata, DecisionMetadata()))
                self.prefill_metadata_queue.append(last_metadata)
                self.prefill_metadata_queue.append((request_key, prefill_last_metadata, DecisionMetadata()))
            else:
                self.prefill_metadata_queue.append((request_key, prefill_first_metadata, DecisionMetadata()))
                self.prefill_metadata_queue.append((request_key, prefill_last_metadata, DecisionMetadata()))
        logger.info(
            f"[layerwiseDisaggregated] prepare_prefill_request request_key {request_key}, rank{self.rank}, "
            f"prefill_metadata_queue {self.prefill_metadata_queue}"
        )

    def prepare_decode_request(self, request_key):
        decode_first_metadata = LwdMetadata(request_key, 0, 0, False, False, False, False, 0, False, 0, 0, 0, 0, False)
        decode_last_metadata = LwdMetadata(request_key, 1, 1, True, False, False, False, 0, False, 0, 0, 0, 0, False)
        self.decode_metadata_queue.append((request_key, decode_first_metadata))
        self.decode_metadata_queue.append((request_key, decode_last_metadata))

    def print_do_inference_log(self, decision_type: DecisionType):
        prefill_metadata = self.prefill_metadata_queue[0][1] if self.prefill_metadata_queue else None
        decode_metadata = self.decode_metadata_queue[0][1] if self.decode_metadata_queue else None
        logger.info(
            f"[layerwiseDisaggregated] decision_type:{decision_type.name}, "
            f"has prefill_first:{prefill_metadata is not None and prefill_metadata.start_exec_layer == 0}, "
            f"has prefill_last:{prefill_metadata is not None and prefill_metadata.end_exec_layer == 1}, "
            f"prefill_comm_finish:{self.prefill_comm_finish}, "
            f"has decode_first:{decode_metadata is not None and decode_metadata.start_exec_layer == 0}, "
            f"has decode_last:{decode_metadata is not None and decode_metadata.end_exec_layer == 1}, "
            f"decode_comm_finish:{self.decode_comm_finish}, "
            f"clean_up_queue size:{self.clean_up_queue.qsize()}, "
            f"clean_eos_queue size:{self.clean_eos_queue.qsize()}."
        )
