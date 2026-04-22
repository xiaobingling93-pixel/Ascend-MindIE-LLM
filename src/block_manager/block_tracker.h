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

#ifndef BLOCK_TRACKER_H
#define BLOCK_TRACKER_H

#include <functional>
#include <limits>
#include <unordered_map>

#include "device_aware_block_allocator.h"
#include "math_utils.h"
#include "sequence.h"

namespace mindie_llm {
// 为 std::pair 提供哈希函数
struct PairHash {
    template <typename T1, typename T2>
    size_t operator()(const std::pair<T1, T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        size_t hashValue = h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << mindie_llm::HASH_SHIFT_LEFT) +
                                 (h1 >> mindie_llm::HASH_SHIFT_RIGHT));
        return hashValue;
    }
};

HashValue ComputeHashValueForSeq(HashValue prevBlockHash, std::vector<TokenId> &tokenIds, HashValue extraHash);
class BlockComputedAttr {
    struct BlockInfo {
        bool active = false;
        bool computed = false;
        TimeStamp lastAccessed = DEFAULT_LAST_ACCESSED_TIME;
    };

   public:
    BlockComputedAttr(size_t numBlocks, BlockId beginBlockId) : blockInfos_(numBlocks), beginBlockId_(beginBlockId) {}

    ~BlockComputedAttr() = default;

    void Reset(BlockId blockId);

    void Enable(BlockId blockId);

    void Disable(BlockId blockId);

    void SetActive(BlockId blockId, bool active);

    bool IsActive(BlockId blockId) const;

    void SetComputed(BlockId blockId, bool computed);

    bool IsComputed(BlockId blockId) const;

    void UpdateAccessTime(BlockId blockId, TimeStamp now);

    TimeStamp LastAccessed(BlockId blockId) const;

   private:
    std::vector<BlockInfo> blockInfos_;

    BlockId beginBlockId_;

    bool IsValidBlockId(BlockId blockId) const {
        if (beginBlockId_ > std::numeric_limits<BlockId>::max() - static_cast<BlockId>(blockInfos_.size())) {
            throw std::runtime_error("blockId range overflow!");
        }
        if (blockId >= beginBlockId_ && blockId < beginBlockId_ + static_cast<BlockId>(blockInfos_.size())) {
            return true;
        } else {
            throw std::runtime_error("blockId is error!");
        }
    }
};

class SeqsBlocksComputedTracker {
   public:
    SeqsBlocksComputedTracker() = default;

    SeqsBlocksComputedTracker(DeviceAwareBlockAllocatorSPtr allocator, size_t blockSize, bool enableCaching,
                              size_t rankSize)
        : allocator_(allocator), blockSize_(blockSize), enableCaching_(enableCaching), rankSize_(rankSize) {}

    ~SeqsBlocksComputedTracker() = default;

    // only computed blocks are considered cached. and for prefill request, just read from recorded cached tokens.
    size_t GetCachedTokensNum(const SequenceSPtr &seq, size_t rankIdx, std::vector<HashValue> &blockHashes,
                              bool seqPrefillFlag);

    size_t GetCachedTokensNum(const SequenceSPtr &seq);

    void RemoveSeq(SequenceId seqId);

   private:
    DeviceAwareBlockAllocatorSPtr allocator_;

    size_t blockSize_;

    bool enableCaching_;

    size_t rankSize_{1};

    std::unordered_map<std::pair<SequenceId, size_t>, size_t, PairHash> seqIdToNumComputedTokens_ = {};
};

class SeqsLastAccessBlocksTracker {
   public:
    SeqsLastAccessBlocksTracker() = default;

    explicit SeqsLastAccessBlocksTracker(DeviceAwareBlockAllocatorSPtr allocator) : allocator_(allocator) {}

    ~SeqsLastAccessBlocksTracker() = default;

    void AddSeq(SequenceId seqId);

    void RemoveSeq(SequenceId seqId);

    void UpdateSeqLastAccess(SequenceId seqId, TimeStamp time);

    void UpdateSeqBlocksLastAccess(SequenceId seqId, std::vector<std::vector<BlockId>> &rankedBlockIds);

   private:
    DeviceAwareBlockAllocatorSPtr allocator_;

    std::unordered_map<SequenceId, TimeStamp> seqIdToLastAccessTime_ = {};
};
}  // namespace mindie_llm

#endif
