#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

# ====================== 1. 加载基础环境 ======================
# 导入 Ascend ATB 和模型依赖环境变量
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/atb-models/set_env.sh
export MINDIE_LOG_TO_STDOUT=1                                  # 控制日志日志打屏

# ====================== 2. 内存管理优化 ======================
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True         # 启用动态显存分段分配，防止一次性大块申请导致OOM
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3                      # 设置 workspace 内存分配算法为策略3（优化复用）
export NPU_MEMORY_FRACTION=0.96                                # 限制单任务最大使用96%显存，保留空间防止爆显存

# ====================== 3. 执行调度与性能优化 ======================
export MINDIE_ASYNC_SCHEDULING_ENABLE=1                        # 启用MindIE异步调度机制
export ATB_OPERATION_EXECUTE_ASYNC=1                           # 所有op异步执行，加速推理
export ATB_LLM_ENABLE_AUTO_TRANSPOSE=0                         # 关闭自动转置优化（某些模型不兼容）

# ====================== 4. 分布式/多卡通信设置 ======================
export WORLD_SIZE=16                                           # 全局设备数为16（用于多卡并发）
export HCCL_CONNECT_TIMEOUT=7200                               # HCCL连接超时时间设为7200秒（适配大模型初始化）
export HCCL_EXEC_TIMEOUT=0                                     # 禁用HCCL执行超时（防止意外中断）
export HCCL_OP_EXPANSION_MODE="AIV"                            # 使用AIV模式展开通信op，提高效率
export HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT=TRUE                # 启用RDMA+PCIE直通，提高传输效率
export ATB_LLM_HCCL_ENABLE=1                                   # 启用ATB LLM通信功能，确保推理多卡可用

# ====================== 5. 数值异常处理设置 ======================
export INF_NAN_MODE_ENABLE=1                                   # 启用NaN/Inf数值检查，便于调试模型异常

# ====================== 6. CPU多线程设置 ======================
export OMP_NUM_THREADS=16                                      # 设置OpenMP使用16个CPU线程

# 遇到config.json文件权限问题
chmod 640 /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
find /usr/local/lib/python3.11/site-packages/mindie* -name config.json | xargs chmod -R 640

# 执行二进制文件，启动服务
/usr/local/Ascend/mindie/latest/mindie-service/bin/mindieservice_daemon
