/*
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

#include <memory>
#include <vector>
#include <queue>
#include <cstddef>

#include "basic_types.h"
#include "device_aware_block_allocator.h"

namespace mindie_llm {

class BlockTable {
public:
    static std::vector<std::vector<TokenId>> ChunkTokensForAllocate(const std::vector<TokenId> &tokenIds,
                                                                    size_t chunkSize);
    BlockTable() = default;

    BlockTable(size_t blockSize, DeviceAwareBlockAllocatorSPtr &blockAllocator, size_t rankSize = 1);

    BlockTable(size_t blockSize, DeviceAwareBlockAllocatorSPtr &blockAllocator,
               std::vector<std::vector<BlockObjSPtr>> &blockObjs, size_t rankSize = 1);

    ~BlockTable() = default;

    void Allocate(const std::vector<TokenId> &tokenIds, DeviceType device = DeviceType::NPU, HashValue extraHash = 0);

    void AllocateSmallRankFirst(const std::vector<TokenId> &tokenIds, DeviceType device = DeviceType::NPU,
                                HashValue extraHash = 0);

    // some block ops needs update block objects directly. convenient but not safe.
    void Update(const std::vector<BlockObjSPtr> &blockObjs);

    // tokenIds 为sequence tokenIds的增量
    void AppendTokenIds(const std::vector<TokenId> &tokenIds, HashValue extraHash = 0, size_t numLookaheadSlots = 0);

    bool CanAppendNewTokens(const std::vector<TokenId> &tokenIds, size_t numLookaheadSlots = 0) const;

    void AppendNewTokens(const std::vector<TokenId> &newTokenIds, HashValue extraHash, size_t numLookaheadSlots);

    void AppendToSpRank(size_t spRank, const std::vector<TokenId>& newTokenIds);

    // ESTIMATE how many blocks needed (maybe bigger than really needed).
    size_t GetNumRelatedBlocks(size_t tokenIdsSize, size_t numLookaheadSlots = 0) const;

    // fork this blocktable, share same block ids.
    BlockTable Fork();

    void Free();

    const std::vector<BlockId> &GetBlockIds() const;

    std::vector<BlockId> GetRankedBlockIds(size_t rankIdx) const;

    std::vector<BlockObjSPtr> GetBlockObjs() const;

    size_t GetNumFullSlots() const;

    std::vector<TokenId> GetNewGenTokenIds(const std::vector<TokenId> &tokenIds) const;

    void ReplaceTrailingPlaceHolder(const std::vector<TokenId> &tokenIds, size_t trailingPlaceHolderNum,
                                    size_t replacedPlaceHolderNum, size_t rankId = 0);

    size_t GetLatestAppendedRankId() const;

    size_t GetAppendedBlockRankId() const;

    bool IsAppendBlock() const;

private:
    void EnsureEnoughSlots(size_t numRequiredEmptySlots, HashValue extraHash = 0, size_t rankId = 0);

    std::vector<BlockObjSPtr> AllocateBlocksForTokenIds(const std::vector<TokenId> &tokenIds, DeviceType device,
                                                        HashValue extraHash);

    std::vector<BlockObjSPtr> AllocateBlocksForTokenIdsSmallRankFirst(const std::vector<TokenId> &tokenIds,
                                                                      DeviceType device, HashValue extraHash);

    size_t GetNumTokenIds() const;

    bool IsAllocated() const;

    size_t GetNumEmptySlots(size_t rankId = 0) const;

    std::vector<std::vector<TokenId>> ChunkTokensForAppend(const std::vector<TokenId> &tokenIds) const;

    // blockObjs_的vector的第一维度是rank
    std::vector<std::vector<BlockObjSPtr>> blockObjs_;

    std::vector<BlockId> blockIds_;

    size_t blockSize_;

    std::vector<size_t> numFullSlotsPerRank_;

    DeviceAwareBlockAllocatorSPtr blockAllocator_;

    // 新增FIFO队列用于存储上一次的rank
    std::queue<size_t> lastUsedRanks_;

    size_t rankSize_ = 1;

    size_t currentSpRank_ = 0;

    bool isAppendBlock_{false};

    size_t appendBlockRankId_ = 0;
};
} // namespace mindie_llm