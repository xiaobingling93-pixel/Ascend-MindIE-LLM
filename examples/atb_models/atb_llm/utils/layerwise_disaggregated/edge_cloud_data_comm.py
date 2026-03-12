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

import os
import queue
import threading

import torch
import torch_npu
from atb_llm.utils.log import logger

EDGE = "master"
CLOUD = "slave"
HCCL = 'hccl'

SEQ_LEN = 32 * 1024
MULTI_NODES_RESV_SEQ_LEN = 1 * 1024 # reserve 1K seq len for malloc memory
BATCH_LEN = 1024

MAX_COMM_GROUPS = 2

SEND_P = 0
RECV_P = 1
SEND_D = 2
RECV_D = 3

P_INDEX = 0
D_INDEX = 1
RECV_INDEX = 0
SEND_INDEX = 1


class EdgeCloudDataComm: 
    def __init__(self, dtype=torch.bfloat16, batch_p_num=1):
        self.role = None
        self.master_ip = None
        self.master_port = None
        self.rank = None
        self.npu_device_id = None
        self.dtype = dtype
        self.edge_ranks_num = None
        self.cloud_ranks_num = None
        self.init_finish = False

        self.group_intra_broadcast_edge = None
        self.map_intra_broadcast_edge = []
        self.groups_intra_broadcast_cloud = [None] * MAX_COMM_GROUPS
        self.maps_intra_broadcast_cloud = [None] * MAX_COMM_GROUPS
        self.groups_inter_send_recv = [None] * MAX_COMM_GROUPS # (comm_group_size, 2), 2: recv_p, 
        self.maps_inter_send_recv = [None] * MAX_COMM_GROUPS

        self.streams = [None] * 2 # [P_stream, D_stream]
        
        self.send_index = 0
        self.recv_index = 0
        self.wait_recv_index = 0
        
        self.cards = [None] * 4  # PD收发时使用的NPU卡号

        self.hidden_size = 7168 # No specific requirements; considering general model settings, use 7168.
        self.batch_p_num = batch_p_num
        self.out_hidden_p = [None] * self.batch_p_num
        self.out_hidden_d = None
        self.target_p = [None] * self.batch_p_num
        self.target_d = None
        self.ret_p = [None] * self.batch_p_num
        self.ret_d = None

        self.prefill_seq_len_queue = queue.Queue()
        self.decode_batch_size_queue = queue.Queue()
        self.p_shape = [None] * self.batch_p_num
        self.d_shape = None

        self.lock = threading.Lock()
        self.flag_pre_recv = True

        self.set_decode_device_done = False
        self.set_prefill_device_done = False

        self.max_input_len = 1
        self.comm_group_size = 1    # 通信域个数, 取cp和dp的最大值
        self.comm_group = 0
        self.multi_nodes_infer_enabled = False
        self.multi_nodes_is_master = False

        self.rank_file = None
        self.npu_id = None
        self.global_comm = None
        self.lwd_rank_file = os.environ.get('LAYERWISE_DISAGGREGATED_RANK_TABLE_FILE')
        self.groups_inter_cleanup = False
        self.npu_net_host = False

    def get_lwd_rank_file(self):
        return self.lwd_rank_file

    def initialize(self, npu_device_id):
        self.npu_device_id = npu_device_id
        logger.info(f"[layerwiseDisaggregated]data comm init npu device is {npu_device_id}")

    def temp_unset_rank_table(self):
        self.rank_file = os.environ.get('RANK_TABLE_FILE')
        os.environ.pop('RANK_TABLE_FILE', None)
        logger.info(f"[layerwiseDisaggregated] remove RANK_TABLE_FILE ENV {self.rank_file}")

    def restore_rank_table(self):
        if self.rank_file is None:
            if self.lwd_rank_file is not None:
                os.environ.pop('RANK_TABLE_FILE', None)
            logger.info("[layerwiseDisaggregated] CONFIRM RANK_TABLE_FILE is None")
            return

        os.environ['RANK_TABLE_FILE'] = self.rank_file
        logger.info(f"[layerwiseDisaggregated] RECOVER RANK_TABLE_FILE: {self.rank_file}")
        self.rank_file = None

    def init_multi_nodes_infer(self, multi_nodes_infer_args):
        if multi_nodes_infer_args is None:
            return
        self.multi_nodes_infer_enabled = True
        self.multi_nodes_is_master = multi_nodes_infer_args['is_master']
        self.comm_group_size = multi_nodes_infer_args['comm_group_size']
        self.max_input_len = multi_nodes_infer_args['max_input_len']
        if self.comm_group_size == 1:
            return
        if self.role == CLOUD:
            self.comm_group = 0 if self.multi_nodes_is_master else 1
        else:
            self.comm_group = self.rank
            
    def init_stream_cards(self):
        if self.comm_group_size == 2: # dp或cp=2 的场景，P/D 的收发均在 0 卡，所有节点都需要要初始化
            if self.role == EDGE:
                self.cards = [self.rank] * 4
            else:
                self.cards = [0] * 4
            p_streams = []
            for _ in range(self.batch_p_num):
                p_streams.append([torch.npu.Stream() for _ in range(2)])
            self.streams[P_INDEX] = p_streams  # streams:[[[P1-recv, P1-send], [P2-recv, P2-send]], [D-recv, D-send]]
            d_streams = [torch.npu.Stream() for _ in range(2)]
            self.streams[D_INDEX] = d_streams
            return

        if self.comm_group_size == 1: # 通信域为1 的场景，0 卡收发 P，1 卡收发 D
            self.cards = [0, 0, 1, 1]
            # 通信域为1 时，CLOUD从节点不需要初始化
            if self.role == CLOUD and self.multi_nodes_infer_enabled and not self.multi_nodes_is_master:
                return
            if self.rank == 0:
                p_streams = []
                for _ in range(self.batch_p_num):
                    p_streams.append([torch.npu.Stream() for _ in range(2)])
                self.streams[P_INDEX] = p_streams
            if self.rank == 1:
                d_streams = [torch.npu.Stream() for _ in range(2)]
                self.streams[D_INDEX] = d_streams

    def final_cleanup(self):
        # NDR场景需要提前释放机间通信域，机内和全局通信域待系统自动释放
        self.groups_inter_cleanup = True
        for groups in self.groups_inter_send_recv:
            if groups is not None:
                for _ in range(self.batch_p_num):
                    if len(groups[P_INDEX]) != 0:
                        torch.distributed.destroy_process_group(group=groups[P_INDEX][0])
                        del groups[P_INDEX][0]
                if len(groups) > 1:
                    torch.distributed.destroy_process_group(group=groups[D_INDEX])
                    del groups[D_INDEX]

    def init_hccl(self):
        self.init_stream_cards()

        self.temp_unset_rank_table()
        if self.lwd_rank_file is not None:
            os.environ['RANK_TABLE_FILE'] = self.lwd_rank_file
        os.environ['MASTER_ADDR'] = self.master_ip
        os.environ['MASTER_PORT'] = str(self.master_port)
        os.environ['WORLD_SIZE'] = str(self.edge_ranks_num + self.cloud_ranks_num)

        edge_rank_offset = 0 if self.lwd_rank_file is None else self.cloud_ranks_num
        cloud_rank_offset = self.edge_ranks_num if self.lwd_rank_file is None else 0

        global_rank = self.rank
        if self.role == CLOUD:
            global_rank += cloud_rank_offset
            if self.multi_nodes_infer_enabled and not self.multi_nodes_is_master:
                global_rank += self.cloud_ranks_num // 2 # slavecount
        else:
            global_rank += edge_rank_offset
        os.environ['RANK'] = str(global_rank)
        torch.distributed.init_process_group(backend=HCCL, init_method='env://')
        logger.info(f"[layerwiseDisaggregated] EdgeCloudDataComm, rank {global_rank} init_process_group")

        if self.comm_group_size == 1:
            # Definition of edge-cloud broadcast group
            self.group_intra_broadcast_edge = torch.distributed.new_group(
                ranks=list(range(edge_rank_offset, edge_rank_offset + self.edge_ranks_num)), backend=HCCL)
            self.map_intra_broadcast_edge = list(range(edge_rank_offset, edge_rank_offset + self.edge_ranks_num))
            self.groups_intra_broadcast_cloud[0] = torch.distributed.new_group(
                ranks=list(range(cloud_rank_offset, cloud_rank_offset + self.cloud_ranks_num)), backend=HCCL)
            self.maps_intra_broadcast_cloud[0] = list(range(cloud_rank_offset,
                                                            cloud_rank_offset + self.cloud_ranks_num))
        else:
            # 通信域为2 场景，边侧不需要创建通信域，云侧需要创建两个通信域
            start_rank = cloud_rank_offset
            rank_per_dp = self.cloud_ranks_num // self.comm_group_size
            for i in range(self.comm_group_size):
                self.groups_intra_broadcast_cloud[i] = torch.distributed.new_group(
                    ranks=list(range(start_rank, start_rank + rank_per_dp)), backend=HCCL)
                self.maps_intra_broadcast_cloud[i] = list(range(start_rank, start_rank + rank_per_dp))
                start_rank += rank_per_dp
        logger.info(f"[layerwiseDisaggregated] EdgeCloudDataComm init braodcast groups: \
                {self.map_intra_broadcast_edge, self.maps_intra_broadcast_cloud}")

        # Definition of inter-node send-recv group
        if self.comm_group_size == 1:
            self.groups_inter_send_recv[0] = []
            self.maps_inter_send_recv[0] = []
            
            # P rank0 收发 P
            p_group = []
            p_map = []
            for _ in range(self.batch_p_num):
                p_group.append(
                        torch.distributed.new_group(ranks=[0 + edge_rank_offset, 0 + cloud_rank_offset], backend=HCCL))
                p_map.append([0 + edge_rank_offset, 0 + cloud_rank_offset])
            self.groups_inter_send_recv[0].append(p_group)
            self.maps_inter_send_recv[0].append(p_map)
            
            # D rank1 收发 D
            self.groups_inter_send_recv[0].append(
                torch.distributed.new_group(ranks=[1 + edge_rank_offset, 1 + cloud_rank_offset], backend=HCCL))
            self.maps_inter_send_recv[0].append([1 + edge_rank_offset, 1 + cloud_rank_offset])    
        else:
            rank_per_dp = self.cloud_ranks_num // self.comm_group_size
            for i in range(self.comm_group_size):
                self.groups_inter_send_recv[i] = []
                self.maps_inter_send_recv[i] = []
                # P
                p_group = []
                pd_map = []
                for _ in range(self.batch_p_num):
                    p_group.append(
                        torch.distributed.new_group(ranks=[i + edge_rank_offset, cloud_rank_offset + i * rank_per_dp],
                                                    backend=HCCL))
                    pd_map.append([i + edge_rank_offset, cloud_rank_offset + i * rank_per_dp])
                self.groups_inter_send_recv[i].append(p_group)
                self.maps_inter_send_recv[i].append(pd_map)
                # D
                self.groups_inter_send_recv[i].append(
                    torch.distributed.new_group(ranks=[i + edge_rank_offset, cloud_rank_offset + i * rank_per_dp],
                                                backend=HCCL))
                self.maps_inter_send_recv[i].append([i + edge_rank_offset, cloud_rank_offset + i * rank_per_dp])

        logger.info(f"[layerwiseDisaggregated] EdgeCloudDataComm init maps_inter_send_recv: "
                    f"{self.maps_inter_send_recv}")

        if self.lwd_rank_file is None:
            torch.distributed.barrier()	 
            data = torch.tensor([0], dtype=torch.float16, device='npu')	 
            torch.distributed.broadcast(data, src=0)
        else:
            if self.role == CLOUD:
                for i in range(self.comm_group_size):
                    torch.distributed.barrier(group=self.groups_intra_broadcast_cloud[i])
            else:
                if self.comm_group_size == 1:
                    torch.distributed.barrier(group=self.group_intra_broadcast_edge)

        torch_npu.npu.synchronize()
        logger.info("[layerwiseDisaggregated] EdgeCloudDataComm: cloud broadcast group init success")

        self.init_finish = True
        # No inter-node communication warmup is performed here; it will be conducted prior to model-level computation.

    def hidden_malloc(self):
        seq_len = SEQ_LEN
        if self.multi_nodes_infer_enabled:
            seq_len = self.max_input_len + MULTI_NODES_RESV_SEQ_LEN
        for i in range(self.batch_p_num):
            if self.out_hidden_p[i] is None:
                self.out_hidden_p[i] = torch.ones((seq_len, self.hidden_size), dtype=self.dtype, device='npu')
        if self.out_hidden_d is None:
            self.out_hidden_d = torch.ones((BATCH_LEN, self.hidden_size), dtype=self.dtype, device='npu')

        logger.info(f"[layerwiseDisaggregated] {torch.distributed.get_rank()}-{self.rank} init max seq len: {seq_len}.")

    def get_global_comm(self):
        default_pg = torch.distributed.group.WORLD
        self.global_comm = default_pg._get_backend(torch.device("npu")).get_hccl_comm(self.npu_id)

    def hccl_comm_warmup(self, hidden_size):
        if hidden_size and hidden_size != self.hidden_size:
            self.hidden_size = hidden_size

        if self.role == EDGE:
            self.warmup_send(1)
            self.warmup_recv(1)
        elif self.multi_nodes_infer_enabled and self.comm_group_size == 1:
            if self.multi_nodes_is_master: # Only master need warm up
                self.warmup_recv(0)
                self.warmup_send(0)
        else:
            self.warmup_recv(0)
            self.warmup_send(0)

        self.restore_rank_table()
        logger.info(f"[layerwiseDisaggregated] EdgeCloudDataComm Warmup send-recv group finish.\
            {torch.distributed.get_rank()} {self.rank}")

        if self.role == CLOUD and self.lwd_rank_file is not None:
            self.get_global_comm()

    def warmup_recv(self, peer_index):
        if self.rank == self.cards[RECV_P]:
            for i in range(self.batch_p_num):
                with torch.npu.stream(self.streams[P_INDEX][i][RECV_INDEX]):
                    ret = torch.distributed.irecv(torch.ones((4096, self.hidden_size), dtype=self.dtype, device='npu'),
                                                group=self.groups_inter_send_recv[self.comm_group][P_INDEX][i],
                                                src=self.maps_inter_send_recv[self.comm_group][P_INDEX][i][peer_index])
                    ret.wait()
                    torch_npu.npu.synchronize()

        if self.rank == self.cards[RECV_D]:
            with torch.npu.stream(self.streams[D_INDEX][RECV_INDEX]):
                ret = torch.distributed.irecv(torch.ones((40, self.hidden_size), dtype=self.dtype, device='npu'),
                                              group=self.groups_inter_send_recv[self.comm_group][D_INDEX],
                                              src=self.maps_inter_send_recv[self.comm_group][D_INDEX][peer_index])
                ret.wait()
                torch_npu.npu.synchronize()

    def warmup_send(self, peer_index):
        if self.rank == self.cards[SEND_P]:
            for i in range(self.batch_p_num):
                with torch.npu.stream(self.streams[P_INDEX][i][SEND_INDEX]):
                    ret = torch.distributed.isend(torch.ones((4096, self.hidden_size), dtype=self.dtype, device='npu'),
                                                group=self.groups_inter_send_recv[self.comm_group][P_INDEX][i],
                                                dst=self.maps_inter_send_recv[self.comm_group][P_INDEX][i][peer_index])
                    ret.wait()
                    torch_npu.npu.synchronize()
                
        if self.rank == self.cards[SEND_D]:
            with torch.npu.stream(self.streams[D_INDEX][SEND_INDEX]):
                ret = torch.distributed.isend(torch.ones((40, self.hidden_size), dtype=self.dtype, device='npu'),
                                              group=self.groups_inter_send_recv[self.comm_group][D_INDEX],
                                              dst=self.maps_inter_send_recv[self.comm_group][D_INDEX][peer_index])
                ret.wait()
                torch_npu.npu.synchronize()

    def broadcast_hidden(self, bc_tensor, shape, mode: str):
        if mode == 'p':
            wait_recv_index = self.wait_recv_index
            self.wait_recv_index = (wait_recv_index + 1) % self.batch_p_num
            
        if self.role == EDGE and self.comm_group_size > 1:  # 多机边侧两卡接收对应云侧的hidden，不需要broadcast
            return bc_tensor
        src_rank = self.cards[RECV_P if mode == 'p' else RECV_D]
        bc_group = self.group_intra_broadcast_edge if self.role == EDGE \
                                                   else self.groups_intra_broadcast_cloud[self.comm_group]
        bc_group_map = self.map_intra_broadcast_edge if self.role == EDGE \
                                                     else self.maps_intra_broadcast_cloud[self.comm_group]

        if bc_tensor is None:
            if mode == 'p':
                self.target_p[wait_recv_index] = self.out_hidden_p[wait_recv_index][:shape[wait_recv_index], :]
                torch.distributed.broadcast(self.target_p[wait_recv_index], src=bc_group_map[src_rank], group=bc_group)
                return self.target_p[wait_recv_index]
            else:
                self.target_d = self.out_hidden_d[:shape, :]
                torch.distributed.broadcast(self.target_d, src=bc_group_map[src_rank], group=bc_group)
                return self.target_d
        else:
            torch.distributed.broadcast(bc_tensor, src=bc_group_map[src_rank], group=bc_group)
            return bc_tensor

    def send_hidden(self, mode: str, out_tensor):
        if self.groups_inter_cleanup:
            return
        peer_index = 1 if self.role == EDGE else 0
        src_rank = self.cards[SEND_P if mode == 'p' else SEND_D]

        if mode == 'p':
            send_index = self.send_index 
            self.send_index = (send_index + 1) % self.batch_p_num

        # dp1 云侧从节点不收发hidden
        is_multi_node_cloud_slave = self.multi_nodes_infer_enabled and self.role == CLOUD and \
            not self.multi_nodes_is_master
        if self.comm_group_size == 1 and is_multi_node_cloud_slave:
            return

        if self.rank == src_rank:
            if mode == 'p':
                p_send_stream = self.streams[P_INDEX][send_index][SEND_INDEX]
                p_send_stream.wait_stream(torch.npu.default_stream())
                with torch.npu.stream(p_send_stream):
                    _ = torch.distributed.isend(
                        tensor=out_tensor, 
                        dst=self.maps_inter_send_recv[self.comm_group][P_INDEX][send_index][peer_index],
                        group=self.groups_inter_send_recv[self.comm_group][P_INDEX][send_index])
            else:
                d_send_stream = self.streams[D_INDEX][SEND_INDEX]
                d_send_stream.wait_stream(torch.npu.default_stream())
                with torch.npu.stream(d_send_stream):
                    _ = torch.distributed.isend(tensor=out_tensor, 
                                                dst=self.maps_inter_send_recv[self.comm_group][D_INDEX][peer_index],
                                                group=self.groups_inter_send_recv[self.comm_group][D_INDEX])

    def recv_hidden(self, mode: str, shape):
        if self.groups_inter_cleanup:
            return
        peer_index = 1 if self.role == EDGE else 0
        src_rank = self.cards[RECV_P if mode == 'p' else RECV_D]

        if mode == 'p':
            recv_index = self.recv_index 
            self.recv_index = (recv_index + 1) % self.batch_p_num

        # dp1 云侧从节点不收发hidden
        is_multi_node_cloud_slave = self.multi_nodes_infer_enabled and self.role == CLOUD and \
            not self.multi_nodes_is_master
        if self.comm_group_size == 1 and is_multi_node_cloud_slave:
            return
            
        if self.rank == src_rank: # 对应的卡才进行收发
            if mode == 'p':
                if self.role == CLOUD and not self.set_prefill_device_done:
                    torch.npu.set_device(torch.device(f"npu:{self.npu_device_id}"))
                    self.set_prefill_device_done = True
                self.target_p[recv_index] = self.out_hidden_p[recv_index][:shape[recv_index], :]
                with torch.npu.stream(self.streams[P_INDEX][recv_index][RECV_INDEX]):
                    ret = torch.distributed.irecv(
                        self.target_p[recv_index], 
                        src=self.maps_inter_send_recv[self.comm_group][P_INDEX][recv_index][peer_index], 
                        group=self.groups_inter_send_recv[self.comm_group][P_INDEX][recv_index])
                self.ret_p[recv_index] = ret
                logger.info(f"[rank-{self.rank}] prefill start async recv, shape={shape[recv_index]}")
            else:
                if self.role == CLOUD and not self.set_decode_device_done:
                    torch.npu.set_device(torch.device(f"npu:{self.npu_device_id}"))
                    self.set_decode_device_done = True
                self.target_d = self.out_hidden_d[:shape, :]
                with torch.npu.stream(self.streams[D_INDEX][RECV_INDEX]):
                    ret = torch.distributed.irecv(self.target_d, 
                                                  src=self.maps_inter_send_recv[self.comm_group][D_INDEX][peer_index],
                                                  group=self.groups_inter_send_recv[self.comm_group][D_INDEX])
                self.ret_d = ret
                logger.info(f"[rank-{self.rank}] decode start async recv, shape={shape}")

    def data_wait_after_recv(self, mode: str):
        src_rank = self.cards[RECV_P if mode == 'p' else RECV_D]

        # dp1 云侧从节点不收发hidden
        is_multi_node_cloud_slave = self.multi_nodes_infer_enabled and self.role == CLOUD and \
            not self.multi_nodes_is_master
        if self.comm_group_size == 1 and is_multi_node_cloud_slave:
            return None

        if self.rank == src_rank:
            if mode == 'p':
                self.ret_p[self.wait_recv_index].wait()
                self.ret_p[self.wait_recv_index] = None
                torch.npu.default_stream().wait_stream(self.streams[P_INDEX][self.wait_recv_index][RECV_INDEX])
                return self.target_p[self.wait_recv_index]
            else:
                self.ret_d.wait()
                torch.npu.default_stream().wait_stream(self.streams[D_INDEX][RECV_INDEX])
                return self.target_d

        return None

    def check_prefill_recv_done(self):
        src_rank = self.cards[RECV_P]

        if self.rank == src_rank:
            if self.ret_p[self.wait_recv_index] is None:
                return True
            return self.ret_p[self.wait_recv_index].is_completed()

        return True

    def npu_net_is_host(self) -> bool:
        return self.npu_net_host

    def npu_net_host_hidden_sync(self, hidden):
        if self.npu_net_host:
            hidden_sync = torch.clone(hidden[0][0])
            hidden_sync.cpu()

    def set_comm_args(self, rank=None, role=None, comm_args=None):
        edge_ip_address = comm_args.get('edge_ip_address', None)
        cloud_ip_address = comm_args.get('cloud_ip_address', None)
        self.master_port = comm_args.get('data_port', None)
        self.edge_ranks_num = comm_args.get('edge_npu_num', None)
        self.cloud_ranks_num = comm_args.get('cloud_npu_num', None)
        multi_nodes_ctrl_port = comm_args.get('multi_nodes_ctrl_port', None)
        multi_nodes_infer_enabled = comm_args.get('multi_nodes_infer_enabled', False)
        multi_nodes_is_master = comm_args.get('multi_nodes_is_master', False)
        comm_group_size = comm_args.get('comm_group_size', None)
        max_input_len = comm_args.get('max_input_len', None)
        self.npu_id = comm_args.get('npu_id', None)
        self.npu_net_host = comm_args.get('npu_net_host', False)

        self.role = role
        self.rank = rank
        master_ip = edge_ip_address if self.lwd_rank_file is None else cloud_ip_address[0]
        self.master_ip = master_ip

        multi_nodes_infer_args = None
        if multi_nodes_infer_enabled:
            multi_nodes_infer_args = {
                'is_master': multi_nodes_is_master,
                'cloud_ctrl_port': multi_nodes_ctrl_port,
                'cloud_ctrl_address': cloud_ip_address[0],
                'comm_group_size': comm_group_size,
                'max_input_len': max_input_len,
            }
        self.init_multi_nodes_infer(multi_nodes_infer_args)