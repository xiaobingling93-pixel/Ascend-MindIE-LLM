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

#include "block_obj.h"

namespace mindie_llm {
enum class BlockAllocatorType : int32_t {
    HASHLESS,
    PREFIXCACHING,
};

class BlockAllocator {
   public:
    BlockAllocator() = default;

    virtual ~BlockAllocator() = default;

    virtual BlockId AllocateBlockId() = 0;

    virtual BlockObjSPtr AllocateMutableBlock(std::vector<TokenId> &tokenIds, BlockObjSPtr prevBlock = nullptr,
                                              HashValue extraHash = 0) = 0;

    virtual BlockObjSPtr AllocateImmutableBlock(std::vector<TokenId> &tokenIds, BlockObjSPtr prevBlock = nullptr,
                                                HashValue extraHash = 0) = 0;

    virtual std::vector<BlockObjSPtr> AllocateImmutableBlocks(std::vector<std::vector<TokenId>> &tokenIds,
                                                              BlockObjSPtr prevBlock = nullptr,
                                                              HashValue extraHash = 0) = 0;

    virtual void FreeBlockId(BlockObjSPtr &block) = 0;

    virtual void Free(BlockObjSPtr &block, bool keepBlockObj) = 0;

    virtual std::vector<BlockObjSPtr> Fork(BlockObjSPtr &lastBlockObj) = 0;

    virtual size_t GetNumTotalBlocks() const = 0;

    virtual size_t GetNumFreeBlock() const = 0;

    virtual std::vector<std::pair<BlockId, BlockId>> ClearCopyOnWrites() = 0;

    virtual void MarkBlocksAsAccessed(const std::vector<BlockId> &blockIds, TimeStamp now) = 0;

    virtual void MarkBlocksAsComputed() = 0;

    virtual std::vector<BlockId> GetCommonComputedBlockIds(
        const std::vector<std::vector<BlockId>> &computedSeqBlockIds) = 0;

    virtual BlockId CowBlockIfNotAppendable(BlockObjSPtr &block) = 0;

    virtual BlockId PromoteToImmutableBlock(const BlockObjSPtr &block) = 0;

    virtual size_t GetNumFullBlocksTouched(const std::vector<BlockObjSPtr> &blocks) = 0;

    virtual void SwapOut(std::vector<BlockObjSPtr> &blocks) = 0;

    virtual void SwapIn(std::vector<BlockObjSPtr> &blocks) = 0;

    virtual float GetPrefixCacheHitRate() const = 0;

    virtual bool ResetPrefixCache() = 0;

    virtual bool FindCachedBlockPrefix(HashValue blockHash) const = 0;

    // 找到最长匹配的前缀的blockid lists
    virtual std::vector<BlockId> FindCachedBlocksPrefix(std::vector<HashValue> &blockHashes) const = 0;

    virtual void AppendTokenIds(BlockObjSPtr blockObj, const std::vector<TokenId> &tokenIds) = 0;

    virtual void ReplaceToken(BlockObjSPtr blockObj, size_t startIndex, TokenId newToken) = 0;

    virtual size_t GetCachedBlockNum(std::vector<HashValue> &hashValues) = 0;
};

using BlockAllocatorPtr = std::unique_ptr<BlockAllocator>;
using BlockAllocatorSPtr = std::shared_ptr<BlockAllocator>;

/*
    A structure to include all params for creating a block object
    Mainly used in BlockFactory
*/
struct BlockSharedAttr {
    size_t blockSize;
    BlockAllocatorSPtr allocator;
};

}  // namespace mindie_llm
