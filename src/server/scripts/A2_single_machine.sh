#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export HCCL_OP_EXPANSION_MODE="AIV"
export NPU_MEMORY_FRACTION=0.92
export ATB_LLM_HCCL_ENABLE=1
export INF_NAN_MODE_ENABLE=1

#8244特性防止oom开关
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=0

#A3到15， A3双机到33
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ATB_OPERATION_EXECUTE_ASYNC=1
export ATB_LLM_ENABLE_AUTO_TRANSPOSE=0
export HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT=TRUE
export HCCL_BUFFSIZE=64

#异步发射
export MINDIE_ASYNC_SCHEDULING_ENABLE=1

#jemalloc优化
export LD_PRELOAD="path_to_file/libjemalloc.so:$LD_PRELOAD"

#队列优化特性
export TASK_QUEUE_ENABLE=1

for var in $(compgen -e | grep 'STDOUT$'); do
    export "$var=0"
done

for var in $(compgen -e | grep 'LOG_TO_FILE$'); do
    export "$var=0"
done

#遇到config.json文件权限问题
find /usr/local/lib/python3.11/site-packages/mindie* -name config.json |xargs chmod -R 640

export MINDIE_LOG_TO_STDOUT=1
export OMP_NUM_THREADS=10
export HCCL_ALGO="level0:NA;level1:pipeline"
