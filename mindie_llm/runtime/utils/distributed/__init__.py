# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import torch.distributed as dist

from mindie_llm.runtime.utils.helpers.env import ENV
from mindie_llm.utils.log.logging import logger
from mindie_llm.runtime.utils.distributed.parallel_info_manager import ParallelInfoManager

_PARALLEL_INFO_MANAGER = None


def set_parallel_info_manager(parallel_info_manager: ParallelInfoManager) -> None:
    """Sets the global parallel info manager instance.

    Args:
        parallel_info_manager (ParallelInfoManager): The parallel info manager instance to set globally.
    """
    global _PARALLEL_INFO_MANAGER
    _PARALLEL_INFO_MANAGER = parallel_info_manager


def get_parallel_info_manager() -> ParallelInfoManager:
    """Retrieves the global parallel info manager instance.

    Returns:
        ParallelInfoManager: The current parallel info manager instance,
        or None if not initialized.
    """
    return _PARALLEL_INFO_MANAGER


def init_distributed(rank: int, world_size: int, local_rank: int, llm_config=None, server_config=None) -> None:
    """Initializes the distributed training environment and parallel info manager.

    This function sets up the PyTorch distributed process group using HCCL backend
    and initializes the global parallel info manager.

    Args:
        rank (int): Global rank of the current process.
        world_size (int): Total number of processes in the distributed setup.
        local_rank (int): The rank (e.g., device index) of the current process.
        llm_config : Configuration for the LLM model. Defaults to None.
        server_config : Configuration for the serving system. Defaults to None.
    """
    if dist.is_initialized():
        return

    master_ip = ENV.master_ip
    if not master_ip:
        raise ValueError("Master IP address is not set, use export MASTER_IP=xxx.xxx.xxx.xxx to solve")
    master_port = ENV.master_port
    if not master_port:
        raise ValueError("Master port is not set, use export MASTER_PORT=xxxx to solve")
    init_method = f"tcp://{master_ip}:{master_port}"
    logger.info(f"rank: {rank}, world_size: {world_size}, init_method: {init_method}, start to init distributed")
    dist.init_process_group(backend="hccl", init_method=init_method, world_size=world_size, rank=rank)

    # initialize parallel info manager
    global _PARALLEL_INFO_MANAGER
    _PARALLEL_INFO_MANAGER = ParallelInfoManager(local_rank, llm_config, server_config)
