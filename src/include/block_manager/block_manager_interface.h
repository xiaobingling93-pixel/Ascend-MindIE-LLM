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
 
#ifndef INTERFACE_H
#define INTERFACE_H
#pragma once

#include <memory>
#include <vector>

#include "basic_types.h"
#include "sequence.h"
#include "sequence_group.h"
namespace mindie_llm {
using SequenceGroupSPtr = std::shared_ptr<SequenceGroup>;
using SequenceSPtr = std::shared_ptr<Sequence>;
enum class BlockManagerType : int32_t {
    SELFATTNBLOCKMANAGER,
    LWDSELFATTNBLOCKMANAGER,
    COMPOSITEBLOCKMANAGER,
    REQUESTSINGLEBLOCKMANAGER,
    REQUESTSLIDINGWINDOWBLOCKMANAGER
};

struct RankedBlockId {
    BlockId blockId;
    size_t rankId;

    bool operator==(const RankedBlockId &other) const { return blockId == other.blockId && rankId == other.rankId; }

    bool operator!=(const RankedBlockId &other) const { return !(*this == other); }
};

enum class RankBlockAllocationMode : int32_t {
    BALANCED,        // 平均分配到各个rank
                     // deprecated
    SMALL_RANK_FIRST // 优先分配到小rank, PD分离场景, P节点使用此模式
};

enum class KvCacheType: int32_t {
    TOKEN = 0,
    SEQUENCE = 1,
    SLIDING_WINDOW = 2
};

struct BlockManagerConfig {
    size_t cacheBlockSize;

    size_t cpuBlockNum;

    size_t npuBlockNum;

    size_t reservedBlockNum; // 预留blocknum 目前固定为0

    size_t speculativeSlots; // 预分配的slot，并行解码场景，会生成多个slot，kvCache存储到预分配的slot中

    bool enableCaching; // 是否开启prefix cache

    size_t rankSize = 1;

    size_t hostSize = 1; // 管理主机的数量，cp场景 hostSize = cpSize

    bool enableKvPool = false;

    std::string cachePoolBackend = "";

    std::string cachePoolConfigPath = "";

    RankBlockAllocationMode allocationMode = RankBlockAllocationMode::SMALL_RANK_FIRST;

    KvCacheType cacheType = KvCacheType::TOKEN;

    // Multi-manager (composite) configs. Only used when BlockManagerType is COMPOSITEBLOCKMANAGER.
    // If empty, COMPOSITEBLOCKMANAGER creation will fail fast.
    std::vector<BlockManagerConfig> subManagers{};

    // Request-scoped window size for request-level block managers (e.g. sliding window manager).
    // Meaning: each request holds at most N BlockIds; when appending beyond N, the oldest block is released.
    size_t requestBlockWindowSize = 2;
};

class BlockSpaceManager {
public:
    BlockSpaceManager() = default;

    virtual ~BlockSpaceManager() = default;

    virtual AllocStatus CanAllocate(const SequenceGroupSPtr &seqGroup) const = 0;

    virtual bool Allocate(const SequenceGroupSPtr &seqGroup) = 0;

    virtual bool CanAppendSlot(const SequenceGroupSPtr &seqGroup) const = 0;

    virtual std::vector<std::pair<BlockId, BlockId>> AppendSlot(const SequenceSPtr &seq) = 0;
    
    // CanAppendSlotNew和AppendSlotNew为sp专用接口，避免对现有流程影响
    virtual bool CanAppendSlotNew(const SequenceGroupSPtr &seqGroup) const = 0;

    virtual void AppendSlotNew(const SequenceGroupSPtr &seqGroup) = 0;

    virtual void AppendTokenToLatestRank(SequenceId seqId, const std::vector<TokenId>& tokens) = 0;
        
    virtual void Fork(SequenceSPtr &parentSeq, SequenceSPtr &childSeq) = 0;

    virtual bool CanSwapOut(const SequenceGroupSPtr &seqGroup) = 0;

    virtual std::vector<std::pair<PhysicalBlockId, PhysicalBlockId>> SwapOut(const SequenceGroupSPtr &seqGroup) = 0;

    virtual AllocStatus CanSwapIn(const SequenceGroupSPtr &seqGroup, size_t numLookheadSlots) = 0;

    virtual std::vector<std::pair<PhysicalBlockId, PhysicalBlockId>> SwapIn(const SequenceGroupSPtr &seqGroup) = 0;

    virtual void Free(SequenceId seqId) = 0;

    virtual std::vector<BlockIds> GetBlockIds(SequenceId seqId) const = 0;

    // virtual std::vector<RankedBlockId> GetRankedBlockIds(SequenceId seqId) const = 0;
    virtual void GetRankedBlockIds(SequenceId seqId, std::vector<RankedBlockId> &rankedBlockIds) const = 0;

    virtual void GetRankedBlockIds(SequenceId seqId, std::vector<std::vector<BlockId>> &rankedBlockIds) const = 0;

    virtual std::vector<std::vector<HashValue>> GetRankedHashValues(SequenceId seqId) const = 0;

    virtual std::vector<HashValue> GetSeqHashValues(SequenceId seqId) const = 0;

    virtual std::vector<size_t> GetTokenCountPerRank(SequenceId seqId) const = 0;

    virtual size_t GetLatestAppendedRankId(SequenceId seqId) const = 0;

    virtual size_t GetAppendedBlockRankId(SequenceId seqId) const = 0;

    virtual bool IsAppendBlock(SequenceId seqId) = 0;

    virtual size_t GetNumFreeNpuBlocks() const = 0;

    virtual size_t GetNumFreeCpuBlocks() const = 0;

    virtual size_t GetTotalNpuBlocks() const = 0;

    virtual void AccessAllblocksInSeq(const SequenceSPtr &seq, float accessTime) = 0;

    virtual std::vector<BlockId> GetCommonComputedBlockIds(const std::vector<SequenceSPtr> &seqs) = 0;

    virtual std::vector<size_t> GetAllrankComputedBlockNum(const std::vector<SequenceSPtr> &seqs) = 0;

    virtual std::vector<BlockId> GetRemoteComputedBlockIds(const std::vector<SequenceSPtr> &seqs,
                                                    size_t computedLens, uint32_t tpSize, std::string modelName) = 0;

    virtual std::vector<size_t> GetAllRankRemoteComputedBlockIds(
        const std::vector<SequenceSPtr> &seqs, std::vector<size_t> &computedBlocksNum, std::string modelName) = 0;

    virtual void MarkBlocksAsComputed() = 0;

    virtual float GetPrefixCacheHitRate() const = 0;

    virtual bool ResetPrefixCache() const = 0;

    virtual size_t GetNumCachedTokens(const SequenceSPtr &seq) = 0;

    virtual size_t GetSeqNumCachedTokens(const SequenceSPtr &seq) = 0;

    virtual void ReplaceTrailingPlaceHolder(const SequenceSPtr &seq, size_t trailingPlaceHolderNum,
                                            size_t replacedPlaceHolderNum) = 0;

    virtual size_t GetLocalDPRank() const = 0;

    virtual std::vector<size_t> GetPrefixBlockOrder(SequenceId seqId, std::vector<size_t> &computedBlocksNum) const = 0;

    virtual void LwdInitCloudBlockManager(const BlockManagerConfig &lwdCloudConfig, size_t localDPRank) = 0;

    virtual void LwdGetCloudRankedBlockIds(SequenceId seqId,
        std::vector<std::vector<BlockId>> &rankedBlockIds) const = 0;

    virtual size_t LwdGetCloudLatestAppendedRankId(SequenceId seqId) const = 0;

    virtual size_t LwdGetCloudAppendedBlockRankId(SequenceId seqId) const = 0;

    virtual std::vector<size_t> LwdGetCloudTokenCountPerRank(SequenceId seqId) const = 0;
};

using BlockSpaceManagerSPtr = std::shared_ptr<BlockSpaceManager>;

// blockmanager 工厂方法
class BlockManagerFactory {
public:
    static BlockSpaceManagerSPtr CreateBlockSpaceManager(BlockManagerType type, const BlockManagerConfig &config,
                                                         size_t localDPRank = 0);
};
}; // namespace mindie_llm
#endif