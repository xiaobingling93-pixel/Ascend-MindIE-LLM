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

#ifndef SELF_ATTN_BLOCK_MANAGER_H
#define SELF_ATTN_BLOCK_MANAGER_H

#include <unordered_map>

#include "basic_types.h"
#include "block_manager_interface.h"
#include "block_table.h"
#include "block_tracker.h"
#include "cpu_npu_block_allocator.h"
#include "mem_pool.h"
#include "sequence.h"
namespace mindie_llm {

/// 此模块负责为自回归生成的token进行kvCache的分配，交换，回收等基础功能，
/// 并提供prefix caching, copy on write等功能的kvCache的分配。
///
/// \param cacheBlockSize   每个Block可以存储的token数量 当前默认为128
/// \param cpuBlockNum      模型侧CacheManager计算得到的CPU上可用Block数量
/// \param npuBlockNum      模型侧CacheManager计算得到的NPU上可用Block数量
/// \param reservedBlockNum 预留blocknum 目前固定为0
/// \param speculativeSlots 预分配的slot，并行解码场景，会生成多个slot，kvCache存储到预分配的slot中
/// \param enableCaching    是否开启prefix cache功能
class SelfAttnBlockManager : public BlockSpaceManager {
   public:
    explicit SelfAttnBlockManager(const BlockManagerConfig &config, size_t localDPRank = 0);

    AllocStatus CanAllocate(const SequenceGroupSPtr &seqGroup) const override;

    bool Allocate(const SequenceGroupSPtr &seqGroup) override;

    bool CanAppendSlot(const SequenceGroupSPtr &seqGroup) const override;

    std::vector<std::pair<BlockId, BlockId>> AppendSlot(const SequenceSPtr &sequence) override;

    // CanAppendSlotNew和AppendSlotNew为sp专用接口，避免对现有流程影响
    bool CanAppendSlotNew(const SequenceGroupSPtr &seqGroup) const override;

    void AppendSlotNew(const SequenceGroupSPtr &seqGroup) override;

    void AppendTokenToLatestRank(SequenceId seqId, const std::vector<TokenId> &tokens) override;

    bool IsAppendBlock(SequenceId seqId) override;

    bool CanSwapOut(const SequenceGroupSPtr &seqGroup) override;

    AllocStatus CanSwapIn(const SequenceGroupSPtr &seqGroup, size_t numLookheadSlots) override;

    std::vector<std::pair<PhysicalBlockId, PhysicalBlockId>> SwapIn(const SequenceGroupSPtr &seqGroup) override;

    std::vector<std::pair<PhysicalBlockId, PhysicalBlockId>> SwapOut(const SequenceGroupSPtr &seqGroup) override;

    void Free(SequenceId seqId) override;

    // REVIEW: need to be changed GetBlocks
    // read block id list from seqId's block Table
    std::vector<BlockIds> GetBlockIds(SequenceId seqId) const override;

    void GetRankedBlockIds(SequenceId seqId, std::vector<RankedBlockId> &rankedBlockIds) const override;

    // rankedBlockIds 第一维度是rank 第二维度是block id
    void GetRankedBlockIds(SequenceId seqId, std::vector<std::vector<BlockId>> &rankedBlockIds) const override;

    std::vector<std::vector<HashValue>> GetRankedHashValues(SequenceId seqId) const override;

    std::vector<HashValue> GetSeqHashValues(SequenceId seqId) const override;

    size_t GetNumFreeNpuBlocks() const override;

    size_t GetNumFreeCpuBlocks() const override;

    size_t GetTotalNpuBlocks() const override { return this->npuBlockNum_; }

    void AccessAllblocksInSeq(const SequenceSPtr &seq, float accessTime) override;

    // REVIEW: mark this round's promoted blocks as computed at the end of this round schedule
    void MarkBlocksAsComputed() override;

    std::vector<BlockId> GetCommonComputedBlockIds(const std::vector<SequenceSPtr> &seqs) override;

    std::vector<size_t> GetAllrankComputedBlockNum(const std::vector<SequenceSPtr> &seqs) override;

    std::vector<BlockId> GetRemoteComputedBlockIds(const std::vector<SequenceSPtr> &seqs, size_t computedLens,
                                                   uint32_t tpSize, std::string modelName) override;

    std::vector<size_t> GetAllRankRemoteComputedBlockIds(const std::vector<SequenceSPtr> &seqs,
                                                         std::vector<size_t> &computedBlocksNum,
                                                         std::string modelName) override;

    void Fork(SequenceSPtr &parentSeq, SequenceSPtr &childSeq) override;

    float GetPrefixCacheHitRate() const override;

    bool ResetPrefixCache() const override;

    // REVIEW: size_t
    // calculate cached and computed token num for this seq.
    size_t GetNumCachedTokens(const SequenceSPtr &seq) override;

    size_t GetSeqNumCachedTokens(const SequenceSPtr &seq) override;

    // 替换 尾部trailingPlaceHolderNum个 中的 前replacedPlaceHolderNum个 为有效值
    void ReplaceTrailingPlaceHolder(const SequenceSPtr &seq, size_t trailingPlaceHolderNum,
                                    size_t replacedPlaceHolderNum) override;
    size_t GetLocalDPRank() const override;

    std::vector<size_t> GetTokenCountPerRank(SequenceId seqId) const override;

    size_t GetLatestAppendedRankId(SequenceId seqId) const override;

    size_t GetAppendedBlockRankId(SequenceId seqId) const override;

    void LwdInitCloudBlockManager(const BlockManagerConfig &lwdCloudConfig, size_t localDPRank = 0) override {};

    void LwdGetCloudRankedBlockIds(SequenceId seqId, std::vector<std::vector<BlockId>> &rankedBlockIds) const override {
    };

    size_t LwdGetCloudLatestAppendedRankId(SequenceId seqId) const override { return 0; };

    size_t LwdGetCloudAppendedBlockRankId(SequenceId seqId) const override { return 0; };

    std::vector<size_t> LwdGetCloudTokenCountPerRank(SequenceId seqId) const override { return {}; };

   private:
    size_t GetNumRequiredBlocks(size_t seqLen, size_t blockSize) const;

    AllocStatus CanSwap(const SequenceGroupSPtr &seqGroup, DeviceType dstDeviceType, SequenceStatus status,
                        size_t numLookaheads = 0);

    size_t blockSize_{0};

    size_t cpuBlockNum_{0};

    size_t npuBlockNum_{0};

    size_t reservedBlockNum_{0};

    size_t speculativeSlots_{0};

    bool enableCaching_;

    size_t localDPRank_{0};

    size_t rankSize_{1};

    size_t hostSize_{1};

    RankBlockAllocationMode allocationMode_{RankBlockAllocationMode::SMALL_RANK_FIRST};

    std::unordered_map<SequenceId, BlockTable> seqId2BlockTable_;

    DeviceAwareBlockAllocatorSPtr blockAllocator_;

    SeqsBlocksComputedTracker seqsBlockComputedTracker_;

    SeqsLastAccessBlocksTracker seqsLastAccessBlocksTracker_;

    MemPoolSPtr memPoolInstance_;
};
using SelfAttnBlockManagerPtr = std::unique_ptr<SelfAttnBlockManager>;
using SelfAttnBlockManagerSPtr = std::shared_ptr<SelfAttnBlockManager>;
}  // namespace mindie_llm

#endif  // SELF_ATTN_BLOCK_MANAGER_H
