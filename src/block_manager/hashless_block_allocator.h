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

#include "block_allocator.h"
#include "block_table.h"
#include "copy_on_write_tracker.h"
#include "hashless_block_obj.h"
#include "obj_pool.h"
#include "ref_counter_protocol.h"

namespace mindie_llm {
class HashLessBlockAllocator : public BlockAllocator, public std::enable_shared_from_this<HashLessBlockAllocator> {
   public:
    HashLessBlockAllocator(BlockId beginBlockId, size_t numBlocks, size_t blockSize, BlockObjPoolSPtr blockObjPool);

    ~HashLessBlockAllocator() override = default;

    BlockId AllocateBlockId() override;

    BlockObjSPtr AllocateMutableBlock(std::vector<TokenId> &tokenIds, BlockObjSPtr prevBlock = nullptr,
                                      HashValue extraHash = 0) override;

    BlockObjSPtr AllocateImmutableBlock(std::vector<TokenId> &tokenIds, BlockObjSPtr prevBlock = nullptr,
                                        HashValue extraHash = 0) override;

    std::vector<BlockObjSPtr> AllocateImmutableBlocks(std::vector<std::vector<TokenId>> &tokenChunks,
                                                      BlockObjSPtr prevBlock = nullptr,
                                                      HashValue extraHash = 0) override;

    void FreeBlockId(BlockObjSPtr &block) override;

    void Free(BlockObjSPtr &block, bool keepBlockObj) override;

    std::vector<BlockObjSPtr> Fork(BlockObjSPtr &lastBlockObj) override;

    size_t GetNumTotalBlocks() const override;

    size_t GetNumFreeBlock() const override;

    std::vector<std::pair<BlockId, BlockId>> ClearCopyOnWrites() override;

    // 这个函数入参不能是const的，因为有可能会free这个block进而reset blockID
    BlockId CowBlockIfNotAppendable(BlockObjSPtr &block) override;

    BlockId PromoteToImmutableBlock(const BlockObjSPtr &block) override;

    void MarkBlocksAsAccessed(const std::vector<BlockId> &blockIds, float now) override;

    void MarkBlocksAsComputed() override;

    std::vector<BlockId> GetCommonComputedBlockIds(const std::vector<std::vector<BlockId>> &id) override;

    size_t GetNumFullBlocksTouched(const std::vector<BlockObjSPtr> &blocks) override;

    void SwapOut(std::vector<BlockObjSPtr> &blocks) override;

    void SwapIn(std::vector<BlockObjSPtr> &blocks) override;

    float GetPrefixCacheHitRate() const override;

    bool ResetPrefixCache() override;

    bool FindCachedBlockPrefix(HashValue blockHash) const override;

    std::vector<BlockId> FindCachedBlocksPrefix(std::vector<HashValue> &blockHashes) const override;

    void AppendTokenIds(BlockObjSPtr blockObj, const std::vector<TokenId> &tokenIds) override;

    void ReplaceToken(BlockObjSPtr blockObj, size_t startIndex, TokenId newToken) override;

    size_t GetCachedBlockNum(std::vector<HashValue> &hashValues) override;

   private:
    size_t blockSize_;

    std::vector<BlockId> allBlockIndices_;

    std::deque<BlockId> freeBlockIndices_;

    RefCounterProtocolSPtr refCounterSPtr_;

    CopyOnWriteTracker cowTracker_;

    BlockObjPoolSPtr blockObjPool_;
};
}  // namespace mindie_llm
