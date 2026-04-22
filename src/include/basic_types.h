/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef BASIC_TYPES_H
#define BASIC_TYPES_H

#include <cstdint>
#include <queue>
#include <string>
#include <vector>
namespace mindie_llm {

// for server module
using RespBodyQueue = std::queue<std::string>;

using TokenId = long;
using Probability = float;
using InstanceId = uint32_t;
using BlockId = int64_t;  // global block id. Like npu block id is from 0~99, cpu block id is from 100~199.
using BlockIds = std::vector<BlockId>;
// SequenceId type need to disscuss
using SequenceId = long;
using RequestId = std::string;
using BatchId = long;
using WaveId = int64_t;

using HashValue = uint64_t;

using TimeStamp = float;
using RefCount = long;
// physical block id is used to determine block physical address. If cpu block id is started from 100, physical block id
// of global block id is 1
using PhysicalBlockId = BlockId;

enum class Role : uint8_t {
    PnD = 0,  // this instance can schedule both P and D
    P,
    D,
    /*
        PD分离特性中的节点角色管理机制。
        在PD分离架构中,不同类型的计算节点被赋予特定角色以优化处理效率:
        Flex(弹性)节点:具备动态任务处理能力,可根据系统负载同时处理Prefill和Decode请求
        角色细分：
        FlexP - 专用于Prefill阶段请求处理
        FlexD - 专用于Decode阶段请求处理
        FlexPnD - 支持Prefill和Decode混合请求的弹性处理
    */
    FlexP,
    FlexD,
    FlexPnD
};

enum class DeviceType : uint8_t {
    CPU,
    NPU,
};

enum class AllocStatus { OK, LATER, NEVER };

// 需要容器IP，建立进程间通信
struct NodeInfo {
    std::string hostIp;
    std::string serviceIp;
};

constexpr BlockId INVALID_BLOCKID = static_cast<BlockId>(-1);
constexpr HashValue INVALID_HASH_VALUE = 0;
constexpr TimeStamp DEFAULT_LAST_ACCESSED_TIME = -1;
constexpr TokenId PLACEHOLDER_TOKEN = -1;
constexpr SequenceId EOS_SEQUENCE_ID = -1;

}  // namespace mindie_llm

#ifdef DEBUG
inline void Assert(bool condition) { assert(condition); }
#else
inline void Assert(bool condition) { (void)condition; }
#endif

#endif
