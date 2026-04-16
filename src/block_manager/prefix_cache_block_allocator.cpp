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

#include "prefix_cache_block_allocator.h"

using namespace std;

namespace mindie_llm {
PrefixCacheBlockAllocator::PrefixCacheBlockAllocator(BlockId beginBlockId, size_t numBlocks, size_t blockSize,
                                                     BlockObjPoolSPtr blockObjPool)
    : beginBlockId_(beginBlockId),
      blockSize_(blockSize),
      numBlocks_(numBlocks),
      blockComputedAttr_(numBlocks_, beginBlockId) {
    if (!blockObjPool) {
        throw std::invalid_argument("blockObjPool cannot be null");
    }

    for (size_t blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
        allBlockIndices_.push_back(blockIdx + beginBlockId);
        freeBlockIndices_.push_back(blockIdx + beginBlockId);
    }

    refCounter_ = MakeRefCounterProtocol(allBlockIndices_);
    if (!refCounter_) {
        throw std::runtime_error("Failed to create refCounter");
    }

    cowTracker_ = CopyOnWriteTracker(refCounter_);
    evictor_ = MakeEvictor(EvictionPolicy::LRU);
    if (!evictor_) {
        throw std::runtime_error("Failed to create evictor");
    }
    hitRateCalculator_ = make_shared<HitRateCalculator>();
    blockObjPool_ = blockObjPool;
}

BlockObjSPtr PrefixCacheBlockAllocator::AllocateMutableBlock(vector<TokenId> &tokenIds, BlockObjSPtr prevBlock,
                                                             HashValue extraHash) {
    BlockId blockId = AllocateBlockId();
    shared_ptr<BlockObj> block = blockObjPool_->AcquireObj();
    BlockAllocatorSPtr allocator = shared_from_this();
    const BlockSharedAttr blockSharedAttr = {blockSize_, allocator};
    std::vector<TokenId> emptyTokenIds;

    block->InitBlockObj(prevBlock, emptyTokenIds, blockSharedAttr, blockId, extraHash);
    AppendTokenIds(block, tokenIds);

    return block;
}

size_t PrefixCacheBlockAllocator::GetCachedBlockNum(std::vector<HashValue> &hashValues) {
    size_t cachedBlockNum = 0;
    for (HashValue key : hashValues) {
        auto it = cachedBlocks_.find(key);
        if (it != cachedBlocks_.end()) {
            cachedBlockNum++;
        } else {
            break;
        }
    }
    return cachedBlockNum;
}

BlockObjSPtr PrefixCacheBlockAllocator::AllocateImmutableBlock(vector<TokenId> &tokenIds, BlockObjSPtr prevBlock,
                                                               HashValue extraHash) {
    BlockId cachedBlockId = INVALID_BLOCKID;
    BlockObjSPtr block = blockObjPool_->AcquireObj();

    BlockAllocatorSPtr allocator = shared_from_this();
    const BlockSharedAttr blockSharedAttr = {blockSize_, allocator};
    block->InitBlockObj(prevBlock, tokenIds, blockSharedAttr, cachedBlockId, extraHash);

    // 优先尝试从cache中找到缓存的blockId
    HashValue key = block->PrefixHash();
    auto it = cachedBlocks_.find(key);
    if (it != cachedBlocks_.end()) {
        cachedBlockId = it->second;
    }

    // 能找到缓存的blockId，直接使用
    if (cachedBlockId != INVALID_BLOCKID) {
        block->SetBlockId(cachedBlockId);
        hitRateCalculator_->Record(true);
        IncrRefCountCacheBlock(block);
        return block;
    }

    // 如果缓存中没有，申请mutable的block, 由于AllocateMutableBlock里面会返回新的block，先释放之前申请的
    blockObjPool_->FreeObj(block);
    hitRateCalculator_->Record(false);

    block = AllocateMutableBlock(tokenIds, prevBlock, extraHash);

    return block;
}

vector<BlockObjSPtr> PrefixCacheBlockAllocator::AllocateImmutableBlocks(vector<vector<TokenId>> &tokenIds,
                                                                        BlockObjSPtr prevBlock, HashValue extraHash) {
    vector<BlockObjSPtr> blockObjs = {};

    BlockObjSPtr frontBlock = prevBlock;
    for (auto tokens : tokenIds) {
        frontBlock = AllocateImmutableBlock(tokens, frontBlock, extraHash);
        blockObjs.push_back(frontBlock);
    }
    return blockObjs;
}

BlockId PrefixCacheBlockAllocator::MayBeAllocateEvictedBlockId() {
    EvictionResult evictionResult = evictor_->Evict();

    BlockId blockId = evictionResult.blockId;
    HashValue key = evictionResult.prefixHash;
    if (blockId == INVALID_BLOCKID) {
        throw runtime_error("BlockAllocator no blockId in evict!");
    }

    // 访问不到就抛出异常
    blockId = cachedBlocks_.at(key);
    if (blockId != evictionResult.blockId) {
        throw runtime_error("BlockAllocator blockId in evict unmatch blockId in cache!");
    }

    if (blockId >= beginBlockId_ + numBlocks_ || blockId < beginBlockId_) {
        throw runtime_error("BlockId out of range!");
    }

    cachedBlocks_.erase(key);
    refCounter_->Increase(blockId);
    TrackBlockId(blockId, false);
    return blockId;
}

BlockId PrefixCacheBlockAllocator::AllocateBlockId() {
    // 如果有空闲的block，先申请空闲的
    if (!freeBlockIndices_.empty()) {
        BlockId blockId = freeBlockIndices_.front();
        freeBlockIndices_.pop_front();
        refCounter_->Increase(blockId);
        TrackBlockId(blockId, false);
        return blockId;
    }

    // 没有空闲的block，使用老化策略获得之前使用过的block
    return MayBeAllocateEvictedBlockId();
}

void PrefixCacheBlockAllocator::FreeBlockId(BlockObjSPtr &block) {
    BlockId blockId = block->GetBlockId();

    if (block->PrefixHash() != INVALID_HASH_VALUE) {
        // blockId没有在使用就加入evictor
        DecrRefCountCacheBlock(block);
    } else {
        RefCount refCount = refCounter_->Decrease(blockId);
        if (refCount == 0) {
            UntrackBlockId(blockId);
            freeBlockIndices_.push_back(blockId);
        }
        block->ResetBlockId();
    }
}

void PrefixCacheBlockAllocator::Free(BlockObjSPtr &block, bool keepBlockObj) {
    FreeBlockId(block);
    if (!keepBlockObj) {
        blockObjPool_->FreeObj(block);
    }
}

vector<BlockObjSPtr> PrefixCacheBlockAllocator::Fork(BlockObjSPtr &lastBlockObj) {
    std::deque<BlockObjSPtr> sourceBlocks = {};
    BlockObjSPtr visitBlockObj = lastBlockObj;
    while (visitBlockObj) {
        sourceBlocks.push_front(visitBlockObj);
        visitBlockObj = visitBlockObj->GetPrevBlock();
    }

    std::vector<BlockObjSPtr> forkedBlocks = {};
    BlockObjSPtr prevBlock = nullptr;

    for (BlockObjSPtr sourceBlock : sourceBlocks) {
        RefCount refCount = refCounter_->Increase(sourceBlock->GetBlockId());
        // 对于未被释放的block，其引用计数至少为1，调用Increase后至少为2
        constexpr long validRefCountThreshold = 2;
        if (refCount < validRefCountThreshold) {
            throw std::runtime_error("can't fork free'd block!");
        }
        std::vector<TokenId> tokenIds = sourceBlock->GetTokenIds();
        BlockObjSPtr forkedBlock = blockObjPool_->AcquireObj();
        BlockAllocatorSPtr allocator = shared_from_this();
        const BlockSharedAttr blockSharedAttr = {blockSize_, allocator};
        forkedBlock->InitBlockObj(prevBlock, tokenIds, blockSharedAttr, sourceBlock->GetBlockId(),
                                  sourceBlock->ExtraHash());
        forkedBlocks.push_back(forkedBlock);
        prevBlock = forkedBlock;
    }

    return forkedBlocks;
}

void PrefixCacheBlockAllocator::IncrRefCountCacheBlock(BlockObjSPtr block) {
    BlockId blockId = block->GetBlockId();
    RefCount refCount = refCounter_->Increase(blockId);
    if (refCount == 1) {
        // 如果在evictor中，需要移除，不再老化
        if (evictor_->ContainsBlock(blockId)) {
            evictor_->Remove(blockId);
        }
        // 重用之后设置为active
        TrackBlockId(blockId, true);
    }
}

void PrefixCacheBlockAllocator::DecrRefCountCacheBlock(BlockObjSPtr block) {
    BlockId blockId = block->GetBlockId();
    block->SetBlockId(INVALID_BLOCKID);
    RefCount refCount = refCounter_->Decrease(blockId);
    if (refCount < 0) {
        throw runtime_error("Prefix blocks blockId counter has been decremented to a negative value.");
    }
    if (refCount == 0) {
        // 没有block使用这个blockID，可以参与老化
        HashValue prefixHash = block->PrefixHash();
        // 严格检查放入evictor中的和cache中的完全一致
        BlockId reuseBlockId;
        auto it = cachedBlocks_.find(prefixHash);
        if (it != cachedBlocks_.end()) {
            reuseBlockId = it->second;
        } else {
            throw runtime_error("Prefix blocks blockId not in cache.");
        }
        if (blockId != reuseBlockId) {
            throw runtime_error("Prefix blocks blockId unmatch.");
        }

        evictor_->Add(blockId, block->PrefixHash(), block->GetNumTokensTotal(),
                      blockComputedAttr_.LastAccessed(blockId));
        UntrackBlockId(blockId);
    }
}

BlockId PrefixCacheBlockAllocator::PromoteToImmutableBlock(const BlockObjSPtr &block) {
    HashValue prefixHash = block->PrefixHash();
    auto it = cachedBlocks_.find(prefixHash);
    // 如果cache中找不到，就将新的插入供后续复用
    if (it == cachedBlocks_.end()) {
        cachedBlocks_[prefixHash] = block->GetBlockId();
        touchedBlocks_.insert(block->GetBlockId());
        return block->GetBlockId();
    } else {
        // 先释放之前的blockId
        RefCount refCount = refCounter_->Decrease(block->GetBlockId());
        if (refCount == 0) {
            UntrackBlockId(block->GetBlockId());
            freeBlockIndices_.push_back(block->GetBlockId());
        }
        block->ResetBlockId();

        // 再使用cached的blockId
        BlockId reuseBlockId = it->second;
        block->SetBlockId(reuseBlockId);
        IncrRefCountCacheBlock(block);
        return reuseBlockId;
    }
}

void PrefixCacheBlockAllocator::MarkBlocksAsAccessed(const vector<BlockId> &blockIds, TimeStamp now) {
    for (BlockId blockId : blockIds) {
        if (blockComputedAttr_.IsActive(blockId)) {
            blockComputedAttr_.UpdateAccessTime(blockId, now);
        } else if (evictor_->ContainsBlock(blockId)) {
            evictor_->Update(blockId, now);
            // 先抛出异常，理论上不应该走到这里，未明确为啥开源软件中这个实现
            throw runtime_error("Block in evictor can not update time.");
        } else {
            throw runtime_error("Mark block as accessed which is not belonged to NPU.");
        }
    }
}

void PrefixCacheBlockAllocator::MarkBlocksAsComputed() {
    for (BlockId blockId : touchedBlocks_) {
        blockComputedAttr_.SetComputed(blockId, true);
    }
    touchedBlocks_.clear();
}

BlockId PrefixCacheBlockAllocator::CowBlockIfNotAppendable(BlockObjSPtr &block) {
    BlockId srcBlockId = block->GetBlockId();
    if (srcBlockId == INVALID_BLOCKID) {
        throw runtime_error("Cow block id invalid.");
    }

    if (cowTracker_.IsAppendable(srcBlockId)) {
        return srcBlockId;
    }

    // 减少一次共享的引用计数
    FreeBlockId(block);
    BlockId trgBlockId = AllocateBlockId();

    cowTracker_.RecordCow(srcBlockId, trgBlockId);

    return trgBlockId;
}

// 在多个BlockIds中找到最长的公共前缀，此时的前缀以blockid为比较对象
vector<BlockId> PrefixCacheBlockAllocator::GetCommonComputedBlockIds(
    const std::vector<std::vector<BlockId>> &computedSeqBlockIds) {
    if (computedSeqBlockIds.empty()) {
        return {};
    }

    if (computedSeqBlockIds.size() == 1) {
        return computedSeqBlockIds[0];
    }

    std::vector<BlockId> commonBlockIds = computedSeqBlockIds[0];

    for (size_t i = 1; i < computedSeqBlockIds.size(); i++) {
        const vector<BlockId> &current = computedSeqBlockIds[i];
        size_t j = 0;

        while (j < commonBlockIds.size() && j < current.size() && current[j] == commonBlockIds[j]) {
            j++;
        }
        commonBlockIds.resize(j);

        if (commonBlockIds.empty()) {
            break;
        }
    }

    return commonBlockIds;
}

size_t PrefixCacheBlockAllocator::GetNumTotalBlocks() const { return allBlockIndices_.size(); }

size_t PrefixCacheBlockAllocator::GetNumFreeBlock() const {
    return freeBlockIndices_.size() + evictor_->GetNumblocks();
}

size_t PrefixCacheBlockAllocator::GetNumFullBlocksTouched(const std::vector<BlockObjSPtr> &blocks) {
    int numTouchedBlocks = 0;

    for (BlockObjSPtr block : blocks) {
        if (!block->IsFull()) {
            continue;
        }
        if (!IsBlockCached(block) ||
            (block->PrefixHash() != INVALID_HASH_VALUE && evictor_->ContainsBlock(block->GetBlockId()))) {
            numTouchedBlocks++;
        }
    }
    return numTouchedBlocks;
}

void PrefixCacheBlockAllocator::SwapOut(std::vector<BlockObjSPtr> &blocks) {
    for (BlockObjSPtr block : blocks) {
        FreeBlockId(block);
    }
}

void PrefixCacheBlockAllocator::SwapIn(std::vector<BlockObjSPtr> &blocks) {
    for (BlockObjSPtr block : blocks) {
        BlockObjSPtr tmpBlock;
        if (block->IsFull()) {
            std::vector<TokenId> token = block->GetTokenIds();
            tmpBlock = AllocateImmutableBlock(token, block->GetPrevBlock(), block->ExtraHash());
        } else {
            std::vector<TokenId> token = block->GetTokenIds();
            tmpBlock = AllocateMutableBlock(token, block->GetPrevBlock(), block->ExtraHash());
        }

        block->SetBlockId(tmpBlock->GetBlockId());
        blockObjPool_->FreeObj(tmpBlock);
    }
}

bool PrefixCacheBlockAllocator::FindCachedBlockPrefix(HashValue blockHash) const {
    if (blockHash == INVALID_HASH_VALUE) {
        return false;
    }
    auto cachedIt = cachedBlocks_.find(blockHash);
    if (cachedIt != cachedBlocks_.end()) {
        return IsBlockComputed(cachedIt->second);
    }
    return false;
}

// 确认实现逻辑是否OK， 遇到不命中时，是否跳过时可行的？
// 找到最长匹配的前缀的blockid lists
std::vector<BlockId> PrefixCacheBlockAllocator::FindCachedBlocksPrefix(std::vector<HashValue> &blockHashes) const {
    std::vector<BlockId> result;
    for (auto blockHash : blockHashes) {
        if (IsBlockCached(blockHash)) {
            result.push_back(cachedBlocks_.at(blockHash));
        } else {
            break;
        }
    }
    return result;
}

float PrefixCacheBlockAllocator::GetPrefixCacheHitRate() const {
    double ret = hitRateCalculator_->GetHitRate();
    // 外部接口之需要float精度
    return static_cast<float>(ret);
}

/* Reset prefix cache. This function may be used in RLHF
flows to invalid prefix caching after the weights are updated,
or used for resetting prefix caching status for benchmarking. */
bool PrefixCacheBlockAllocator::ResetPrefixCache() {
    int useBlockNum = static_cast<int>(GetNumTotalBlocks()) - static_cast<int>(GetNumFreeBlock());
    if (useBlockNum > 0) {
        return false;
    }

    allBlockIndices_.clear();
    freeBlockIndices_.clear();
    for (size_t blockIdx = 0; blockIdx < numBlocks_; blockIdx++) {
        allBlockIndices_.push_back(blockIdx + beginBlockId_);
        freeBlockIndices_.push_back(blockIdx + beginBlockId_);
    }

    refCounter_ = MakeRefCounterProtocol(allBlockIndices_);
    cowTracker_ = CopyOnWriteTracker(refCounter_);
    evictor_ = MakeEvictor(EvictionPolicy::LRU);

    return true;
}

// PrefixCacheBlockAllocator专有方法
bool PrefixCacheBlockAllocator::IsBlockCached(const BlockObjSPtr &block) const {
    HashValue prefixHash = block->PrefixHash();
    if (prefixHash == INVALID_HASH_VALUE) {
        return false;
    }
    auto it = cachedBlocks_.find(prefixHash);
    if (it != cachedBlocks_.end()) {
        return true;
    }
    return false;
}
void PrefixCacheBlockAllocator::TrackBlockId(BlockId blockId, bool computed)

{
    blockComputedAttr_.Enable(blockId);
    blockComputedAttr_.SetComputed(blockId, computed);
}

void PrefixCacheBlockAllocator::UntrackBlockId(BlockId blockId) { blockComputedAttr_.Disable(blockId); }

bool PrefixCacheBlockAllocator::IsBlockComputed(BlockId blockId) const {
    if (blockComputedAttr_.IsActive(blockId)) {
        return blockComputedAttr_.IsComputed(blockId);
    } else {
        return evictor_->ContainsBlock(blockId);
    }
}

bool PrefixCacheBlockAllocator::IsBlockCached(const HashValue prefixHash) const {
    auto it = cachedBlocks_.find(prefixHash);
    if (it == cachedBlocks_.end()) {
        return false;
    }

    BlockId cachedBlockId = it->second;

    // We only consider the blocks that are marked as computed.

    return IsBlockComputed(cachedBlockId);
}

std::vector<std::pair<BlockId, BlockId>> PrefixCacheBlockAllocator::ClearCopyOnWrites() {
    std::vector<std::pair<BlockId, BlockId>> ret = cowTracker_.ClearCows();
    return ret;
}

void PrefixCacheBlockAllocator::AppendTokenIds(BlockObjSPtr blockObj, const std::vector<TokenId> &tokenIds) {
    blockObj->AppendTokenIds(tokenIds);
    BlockId blockId = CowBlockIfNotAppendable(blockObj);  // CoW
    blockObj->SetBlockId(blockId);
    // Append后PrefixHash如果有效，说明block变为full， 并且没有place holder
    if (blockObj->PrefixHash() != INVALID_HASH_VALUE) {
        PromoteToImmutableBlock(blockObj);
        // promote后，可能会与别的block共享block id
    }
}
void PrefixCacheBlockAllocator::ReplaceToken(BlockObjSPtr blockObj, size_t startIndex, TokenId newToken) {
    blockObj->ReplaceToken(startIndex, newToken);
    // 替换后 block 中的place holder会被替换为有效token，此时的PrefixHash 可能会生效，需要走promote流程
    if (blockObj->PrefixHash() != INVALID_HASH_VALUE) {
        PromoteToImmutableBlock(blockObj);
    }
}

}  // namespace mindie_llm
