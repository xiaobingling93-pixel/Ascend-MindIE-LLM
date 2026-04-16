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

#pragma once

#include <unordered_map>

#include "basic_types.h"
#include "block_manager_interface.h"
#include "block_table.h"
#include "block_tracker.h"
#include "cpu_npu_block_allocator.h"
#include "mem_pool.h"
#include "self_attn_block_manager.h"
#include "sequence.h"
namespace mindie_llm {

/// 此模块为边云特性专用的block管理模块
/// 暂不支持prefix caching, copy on write等功能的kvCache的分配。
///
/// \param cacheBlockSize   每个Block可以存储的token数量 当前默认为128
/// \param cpuBlockNum      模型侧CacheManager计算得到的CPU上可用Block数量
/// \param npuBlockNum      模型侧CacheManager计算得到的NPU上可用Block数量
/// \param reservedBlockNum 预留blocknum 目前固定为0
/// \param speculativeSlots 预分配的slot，并行解码场景，会生成多个slot，kvCache存储到预分配的slot中
/// \param enableCaching    是否开启prefix cache功能
class LwdSelfAttnBlockManager : public SelfAttnBlockManager {
   public:
    explicit LwdSelfAttnBlockManager(const BlockManagerConfig &config, size_t localDPRank = 0);

    void LwdInitCloudBlockManager(const BlockManagerConfig &lwdCloudConfig, size_t localDPRank = 0) override;

    void LwdGetCloudRankedBlockIds(SequenceId seqId, std::vector<std::vector<BlockId>> &rankedBlockIds) const override;

    void AccessAllblocksInSeq(const SequenceSPtr &seq, float accessTime) override;

    void Free(SequenceId seqId) override;

    AllocStatus CanAllocate(const SequenceGroupSPtr &seqGroup) const override;

    bool Allocate(const SequenceGroupSPtr &seqGroup) override;

    bool CanAppendSlot(const SequenceGroupSPtr &seqGroup) const override;

    bool CanAppendSlotNew(const SequenceGroupSPtr &seqGroup) const override;

    void AppendSlotNew(const SequenceGroupSPtr &seqGroup) override;

    size_t LwdGetCloudLatestAppendedRankId(SequenceId seqId) const override;

    size_t LwdGetCloudAppendedBlockRankId(SequenceId seqId) const override;

    std::vector<size_t> LwdGetCloudTokenCountPerRank(SequenceId seqId) const override;

   private:
    BlockSpaceManagerSPtr lwdCloudBlockManager_;
};
using SelfAttnBlockManagerPtr = std::unique_ptr<SelfAttnBlockManager>;
using SelfAttnBlockManagerSPtr = std::shared_ptr<SelfAttnBlockManager>;
}  // namespace mindie_llm
