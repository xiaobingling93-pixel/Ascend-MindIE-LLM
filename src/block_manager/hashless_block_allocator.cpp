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

#include "hashless_block_allocator.h"

using namespace std;

namespace mindie_llm {

HashLessBlockAllocator::HashLessBlockAllocator(BlockId beginBlockId, size_t numBlocks, size_t blockSize,
                                               BlockObjPoolSPtr blockObjPool)
    : blockSize_(blockSize), blockObjPool_(blockObjPool) {
    for (size_t blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
        allBlockIndices_.push_back(blockIdx + beginBlockId);
        freeBlockIndices_.push_back(blockIdx + beginBlockId);
    }

    refCounterSPtr_ = MakeRefCounterProtocol(allBlockIndices_);
    cowTracker_ = CopyOnWriteTracker(refCounterSPtr_);
    blockObjPool_ = blockObjPool;
}

BlockId HashLessBlockAllocator::AllocateBlockId() {
    if (freeBlockIndices_.empty()) {
        throw std::runtime_error("BlockAllocator Error: No Free Blocks!");
    }
    BlockId allocatedBlockId = freeBlockIndices_.front();
    freeBlockIndices_.pop_front();
    refCounterSPtr_->Increase(allocatedBlockId);
    return allocatedBlockId;
}

// HashValue型入参extraHash只有虚基类BlockAllocator的其他派生类的同名虚函数会用到
// 在此处参数列表中省略该入参变量名，避免Unused Parameter告警
BlockObjSPtr HashLessBlockAllocator::AllocateMutableBlock(std::vector<TokenId> &tokenIds, BlockObjSPtr prevBlock,
                                                          [[maybe_unused]] HashValue extraHash) {
    BlockId newBlockId = AllocateBlockId();
    BlockObjSPtr mutableHashLessBlockObj = blockObjPool_->AcquireObj();
    BlockAllocatorSPtr blockAllocatorPtr = shared_from_this();
    mutableHashLessBlockObj->InitBlockObj(prevBlock, tokenIds, BlockSharedAttr{blockSize_, blockAllocatorPtr},
                                          newBlockId);
    return mutableHashLessBlockObj;
}

// HashValue型入参extraHash只有虚基类BlockAllocator的其他派生类的同名虚函数会用到
// 在此处参数列表中省略该入参变量名，避免Unused Parameter告警
BlockObjSPtr HashLessBlockAllocator::AllocateImmutableBlock(std::vector<TokenId> &tokenIds, BlockObjSPtr prevBlock,
                                                            [[maybe_unused]] HashValue extraHash) {
    std::vector<TokenId> emptyTokenIds = {};
    BlockObjSPtr immutableHashLessBlockObj = AllocateMutableBlock(emptyTokenIds, prevBlock);
    AppendTokenIds(immutableHashLessBlockObj, tokenIds);
    return immutableHashLessBlockObj;
}

// HashValue型入参extraHash只有虚基类BlockAllocator的其他派生类的同名虚函数会用到
// 在此处参数列表中省略该入参变量名，避免Unused Parameter告警
std::vector<BlockObjSPtr> HashLessBlockAllocator::AllocateImmutableBlocks(
    std::vector<std::vector<TokenId>> &tokenChunks, BlockObjSPtr prevBlock, [[maybe_unused]] HashValue extraHash) {
    size_t numBlocks = tokenChunks.size();

    std::vector<BlockId> allocatedBlockIds = {};
    for (size_t blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
        BlockId allocatedBlockId = AllocateBlockId();
        allocatedBlockIds.push_back(allocatedBlockId);
    }

    std::vector<BlockObjSPtr> immutableHashLessBlockObjs = {};
    BlockAllocatorSPtr blockAllocatorPtr = shared_from_this();
    for (size_t blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
        BlockObjSPtr immutableHashLessBlockObj = blockObjPool_->AcquireObj();
        immutableHashLessBlockObj->InitBlockObj(prevBlock, tokenChunks[blockIdx],
                                                BlockSharedAttr{blockSize_, blockAllocatorPtr},
                                                allocatedBlockIds[blockIdx]);
        immutableHashLessBlockObjs.push_back(immutableHashLessBlockObj);
        prevBlock = immutableHashLessBlockObj;
    }
    return immutableHashLessBlockObjs;
}

void HashLessBlockAllocator::FreeBlockId(BlockObjSPtr &block) {
    BlockId blockId = block->GetBlockId();
    RefCount refCount = refCounterSPtr_->Decrease(blockId);
    if (refCount == 0) {
        freeBlockIndices_.push_back(blockId);
    }
    block->ResetBlockId();
}

void HashLessBlockAllocator::Free(BlockObjSPtr &block, bool keepBlockObj) {
    FreeBlockId(block);
    if (!keepBlockObj) {
        blockObjPool_->FreeObj(block);
    }
}

std::vector<BlockObjSPtr> HashLessBlockAllocator::Fork(BlockObjSPtr &lastBlockObj) {
    std::deque<BlockObjSPtr> sourceBlocks = {};
    BlockObjSPtr visitBlockObj = lastBlockObj;
    while (visitBlockObj) {
        sourceBlocks.push_front(visitBlockObj);
        visitBlockObj = visitBlockObj->GetPrevBlock();
    }

    std::vector<BlockObjSPtr> forkedBlocks = {};
    BlockObjSPtr prevBlock = nullptr;
    BlockAllocatorSPtr blockAllocatorPtr = shared_from_this();

    for (BlockObjSPtr sourceBlock : sourceBlocks) {
        RefCount refCount = refCounterSPtr_->Increase(sourceBlock->GetBlockId());
        // 对于未被释放的block，其引用计数至少为1，调用Increase后至少为2
        constexpr long validRefCountThreshold = 2;
        if (refCount < validRefCountThreshold) {
            throw std::runtime_error("can't fork free'd block!");
        }
        std::vector<TokenId> tokenIds = sourceBlock->GetTokenIds();
        BlockObjSPtr forkedBlock = blockObjPool_->AcquireObj();
        forkedBlock->InitBlockObj(prevBlock, tokenIds, BlockSharedAttr{blockSize_, blockAllocatorPtr},
                                  sourceBlock->GetBlockId());
        forkedBlocks.push_back(forkedBlock);
        prevBlock = forkedBlock;
    }

    return forkedBlocks;
}

size_t HashLessBlockAllocator::GetNumTotalBlocks() const { return allBlockIndices_.size(); }

size_t HashLessBlockAllocator::GetNumFreeBlock() const { return freeBlockIndices_.size(); }

std::vector<std::pair<BlockId, BlockId>> HashLessBlockAllocator::ClearCopyOnWrites() {
    std::vector<std::pair<BlockId, BlockId>> ret = cowTracker_.ClearCows();
    return ret;
}

BlockId HashLessBlockAllocator::CowBlockIfNotAppendable(BlockObjSPtr &block) {
    BlockId srcBlockId = block->GetBlockId();
    if (cowTracker_.IsAppendable(srcBlockId)) {
        return srcBlockId;
    }

    FreeBlockId(block);
    BlockId trgBlockId = AllocateBlockId();
    cowTracker_.RecordCow(srcBlockId, trgBlockId);
    return trgBlockId;
}

// std::vector<std::vector<BlockId>> &型入参只有虚基类BlockAllocator的其他派生类的同名虚函数会用到
// 在此处参数列表中省略该入参变量名，避免Unused Parameter告警
std::vector<BlockId> HashLessBlockAllocator::GetCommonComputedBlockIds(
    [[maybe_unused]] const std::vector<std::vector<BlockId>> &id) {
    return std::vector<BlockId>();
}

size_t HashLessBlockAllocator::GetNumFullBlocksTouched(const std::vector<BlockObjSPtr> &blocks) {
    size_t numFullBlocks = 0;
    for (BlockObjSPtr block : blocks) {
        if (block->IsFull()) {
            numFullBlocks++;
        }
    }
    return numFullBlocks;
}

void HashLessBlockAllocator::SwapOut(std::vector<BlockObjSPtr> &blocks) {
    // 对于换出目标block，释放并重置其blockId，但是保留其原有block对象
    for (BlockObjSPtr block : blocks) {
        FreeBlockId(block);
    }
}

void HashLessBlockAllocator::SwapIn(std::vector<BlockObjSPtr> &blocks) {
    // 分配新的Immutable或Mutable Block对象并取其blockId
    for (BlockObjSPtr block : blocks) {
        if (block == nullptr) {
            throw std::invalid_argument("SwapIn Error: encountered null block.");
        }
        BlockObjSPtr tmpBlock = nullptr;
        if (block->IsFull()) {
            std::vector<TokenId> tokenIds = block->GetTokenIds();
            tmpBlock = AllocateImmutableBlock(tokenIds, block->GetPrevBlock());
        } else {
            std::vector<TokenId> emptyTokenIds = {};
            tmpBlock = AllocateMutableBlock(emptyTokenIds, block->GetPrevBlock());
            AppendTokenIds(tmpBlock, block->GetTokenIds());
        }

        block->SetBlockId(tmpBlock->GetBlockId());
        tmpBlock->ResetBlockId();
        blockObjPool_->FreeObj(tmpBlock);
    }
}

float HashLessBlockAllocator::GetPrefixCacheHitRate() const { return -1; }

bool HashLessBlockAllocator::ResetPrefixCache() { return true; }

bool HashLessBlockAllocator::FindCachedBlockPrefix([[maybe_unused]] HashValue blockHash) const { return false; }

// std::vector<HashValue> &型入参只有虚基类BlockAllocator的其他派生类的同名虚函数会用到
// 在此处参数列表中省略该入参变量名，避免Unused Parameter告警
std::vector<BlockId> HashLessBlockAllocator::FindCachedBlocksPrefix(
    [[maybe_unused]] std::vector<HashValue> &blockHashes) const {
    return std::vector<BlockId>();
}

// 在hashless block中函数不使用
BlockId HashLessBlockAllocator::PromoteToImmutableBlock([[maybe_unused]] const BlockObjSPtr &block) {
    throw std::runtime_error("There is no promotion for naive blocks");
}

size_t HashLessBlockAllocator::GetCachedBlockNum([[maybe_unused]] std::vector<HashValue> &hashValues) {
    throw std::runtime_error("There is no block cached info for naive blocks");
}

// 在hashless block中函数不使用
void HashLessBlockAllocator::MarkBlocksAsAccessed([[maybe_unused]] const std::vector<BlockId> &blockIds,
                                                  [[maybe_unused]] float now) {
    return;
}

// 在hashless block中函数不使用
void HashLessBlockAllocator::MarkBlocksAsComputed() { return; }

void HashLessBlockAllocator::AppendTokenIds(BlockObjSPtr blockObj, const std::vector<TokenId> &tokenIds) {
    blockObj->AppendTokenIds(tokenIds);
    BlockId blockId = CowBlockIfNotAppendable(blockObj);  // CoW
    blockObj->SetBlockId(blockId);
}

void HashLessBlockAllocator::ReplaceToken(BlockObjSPtr blockObj, size_t startIndex, TokenId newToken) {
    blockObj->ReplaceToken(startIndex, newToken);
}

}  // namespace mindie_llm
