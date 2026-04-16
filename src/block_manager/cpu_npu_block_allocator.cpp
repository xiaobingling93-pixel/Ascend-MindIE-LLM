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

#include "cpu_npu_block_allocator.h"

#include "prefix_cache_block.h"
#include "prefix_cache_block_allocator.h"

namespace mindie_llm {
CpuNpuBlockAllocator::CpuNpuBlockAllocator(const AllocatorConfig &config)
    : rankSize_(config.rankSize), beginCpuBlockId_(config.numNpuBlocks), hostSize_(config.hostSize) {
    if (config.rankSize == 0) {
        throw std::runtime_error("Rank size cannot be zero.");
    }

    BlockAllocatorSPtr cpuBlockAllocator;
    BlockAllocatorSPtr npuBlockAllocator;
    BlockId beginNpuBlockId = 0;
    uint32_t extraFactor = 4;

    BlockObjPoolSPtr blockObjPool;
    if (config.allocatorType == BlockAllocatorType::HASHLESS) {
        blockObjPool = std::make_shared<ObjPool<BlockObj>>(
            extraFactor * (config.numCpuBlocks * hostSize_ + config.numNpuBlocks * config.rankSize),
            []() { return std::make_shared<HashLessBlockObj>(); });
    } else if (config.allocatorType == BlockAllocatorType::PREFIXCACHING) {
        blockObjPool = std::make_shared<ObjPool<BlockObj>>(
            extraFactor * (config.numCpuBlocks * hostSize_ + config.numNpuBlocks * config.rankSize),
            []() { return std::make_shared<PrefixCachingBlockObj>(); });
    }

    // CPU allocators为hostSize_
    for (size_t i = 0; i < hostSize_; ++i) {
        if (config.allocatorType == BlockAllocatorType::HASHLESS) {
            cpuAllocators_.emplace_back(std::make_shared<HashLessBlockAllocator>(
                beginCpuBlockId_ + (config.numCpuBlocks) * i, config.numCpuBlocks, config.blockSize, blockObjPool));
        } else if (config.allocatorType == BlockAllocatorType::PREFIXCACHING) {
            cpuAllocators_.emplace_back(std::make_shared<PrefixCacheBlockAllocator>(
                beginCpuBlockId_ + (config.numCpuBlocks) * i, config.numCpuBlocks, config.blockSize, blockObjPool));
        }
    }

    // NPU allocators数量为rankSize_
    for (size_t i = 0; i < config.rankSize; ++i) {
        if (config.allocatorType == BlockAllocatorType::HASHLESS) {
            npuAllocators_.emplace_back(std::make_shared<HashLessBlockAllocator>(beginNpuBlockId, config.numNpuBlocks,
                                                                                 config.blockSize, blockObjPool));
        } else if (config.allocatorType == BlockAllocatorType::PREFIXCACHING) {
            npuAllocators_.emplace_back(std::make_shared<PrefixCacheBlockAllocator>(
                beginNpuBlockId, config.numNpuBlocks, config.blockSize, blockObjPool));
        }
    }

    swapMapping_ = {};
}

DeviceType CpuNpuBlockAllocator::GetDeviceTypeForBlockId(BlockId blockId) const {
    return (blockId < beginCpuBlockId_) ? DeviceType::NPU : DeviceType::CPU;
}

BlockObjSPtr CpuNpuBlockAllocator::AllocateMutableBlock(DeviceType deviceType, std::vector<TokenId> &tokenIds,
                                                        BlockObjSPtr prevBlock, HashValue extraHash, size_t rankIdx) {
    BlockObjSPtr block = GetAllocator(deviceType, rankIdx)->AllocateMutableBlock(tokenIds, prevBlock, extraHash);
    block->SetRankIdx(rankIdx);
    return block;
}

BlockObjSPtr CpuNpuBlockAllocator::AllocateImmutableBlock(DeviceType deviceType, std::vector<TokenId> &tokenIds,
                                                          BlockObjSPtr prevBlock, HashValue extraHash, size_t rankIdx) {
    BlockObjSPtr block = GetAllocator(deviceType, rankIdx)->AllocateImmutableBlock(tokenIds, prevBlock, extraHash);
    block->SetRankIdx(rankIdx);
    return block;
}

std::vector<BlockObjSPtr> CpuNpuBlockAllocator::AllocateImmutableBlocks(DeviceType deviceType,
                                                                        std::vector<std::vector<TokenId>> &tokenIds,
                                                                        BlockObjSPtr prevBlock, HashValue extraHash,
                                                                        size_t rankIdx) {
    std::vector<BlockObjSPtr> blocks =
        GetAllocator(deviceType, rankIdx)->AllocateImmutableBlocks(tokenIds, prevBlock, extraHash);
    for (BlockObjSPtr &block : blocks) {
        block->SetRankIdx(rankIdx);
    }
    return blocks;
}

void CpuNpuBlockAllocator::Free(BlockObjSPtr &block) {
    size_t rankIdx = block->GetRankIdx();
    BlockId blockId = block->GetBlockId();
    bool keepBlockObj = false;
    DeviceType deviceType = GetDeviceTypeForBlockId(blockId);

    GetAllocator(deviceType, rankIdx)->Free(block, keepBlockObj);

    if (block != nullptr) {
        throw std::invalid_argument("Free Block must be nullptr");
    }
}

std::vector<BlockObjSPtr> CpuNpuBlockAllocator::Fork(BlockObjSPtr &block) {
    BlockId blockId = block->GetBlockId();
    DeviceType deviceType = GetDeviceTypeForBlockId(blockId);
    size_t rankIdx = block->GetRankIdx();

    return GetAllocator(deviceType, rankIdx)->Fork(block);
}

size_t CpuNpuBlockAllocator::GetNumTotalBlocks(DeviceType deviceType) const {
    size_t numTotalBlocks = 0;
    for (size_t rankId = 0; rankId < rankSize_; rankId++) {
        numTotalBlocks += GetAllocator(deviceType, rankId)->GetNumTotalBlocks();
    }
    return numTotalBlocks;
}

size_t CpuNpuBlockAllocator::GetNumFreeBlock(DeviceType deviceType) const {
    if (rankSize_ == 1) {
        return GetAllocator(deviceType)->GetNumFreeBlock();
    } else {
        size_t deviceNum = (deviceType == DeviceType::NPU) ? rankSize_ : hostSize_;
        size_t minFreeBlock = GetAllocator(deviceType)->GetNumFreeBlock();
        // 保守计算以负载最重的rank为准
        for (size_t i = 0; i < deviceNum; ++i) {
            size_t freeBlock = GetAllocator(deviceType, i)->GetNumFreeBlock();
            if (freeBlock < minFreeBlock) {
                minFreeBlock = freeBlock;
            }
        }
        return minFreeBlock * rankSize_;
    }
}

size_t CpuNpuBlockAllocator::GetNumFreeBlock(DeviceType deviceType, size_t rankId) const {
    return GetAllocator(deviceType, rankId)->GetNumFreeBlock();
}

PhysicalBlockId CpuNpuBlockAllocator::GetPhysicalBlockId(BlockId globalBlockId) const {
    DeviceType deviceType = GetDeviceTypeForBlockId(globalBlockId);
    if (deviceType == DeviceType::NPU) {
        return globalBlockId;
    } else if (deviceType == DeviceType::CPU) {
        return globalBlockId - beginCpuBlockId_;
    } else {
        throw std::invalid_argument("Unsupported deviceType!");
    }
}

std::vector<std::pair<BlockId, BlockId>> CpuNpuBlockAllocator::Swap(std::vector<BlockObjSPtr> &swapTargetBlocks,
                                                                    DeviceType srcDevice, DeviceType dstDevice) {
    // 对sp， cp暂时不支持swap,走recomputed 或者 cancel
    if (rankSize_ != 1) {
        throw std::invalid_argument("Can not swap when rank size larger than 1!");
    }
    // swapIn会修改swapTargetBlocks中所有block原有blockId
    std::vector<BlockId> srcBlockIds = {};
    for (const auto &srcBlock : swapTargetBlocks) {
        BlockId srcBlockId = srcBlock->GetBlockId();
        DeviceType deviceType = GetDeviceTypeForBlockId(srcBlockId);
        if (deviceType != srcDevice) {
            throw std::invalid_argument("BlockIds in swapTargetBlocks must belong to srcDevice!");
        }
        srcBlockIds.push_back(srcBlockId);
    }

    // SwapIn()采用保留原有Block对象替换新分配的BlockId的方式
    GetAllocator(srcDevice)->SwapOut(swapTargetBlocks);  // free block ids owned
    // allocate new block ids in dst device, in-place block id update
    GetAllocator(dstDevice)->SwapIn(swapTargetBlocks);

    std::vector<BlockId> dstBlockIds = {};
    for (const auto &dstBlock : swapTargetBlocks) {
        dstBlockIds.push_back(dstBlock->GetBlockId());
    }

    if (srcBlockIds.size() != dstBlockIds.size()) {
        throw std::invalid_argument("Length of srcBlockIds and dstBlockIds must be equal!");
    }

    size_t blockIdVecLen = srcBlockIds.size();
    std::vector<std::pair<BlockId, BlockId>> currentSwapMapping = {};
    for (size_t blockIdIdx = 0; blockIdIdx < blockIdVecLen; blockIdIdx++) {
        BlockId srcBlockId = srcBlockIds[blockIdIdx];
        BlockId dstBlockId = dstBlockIds[blockIdIdx];
        // 外部对于swapMapping_表格内容暂时无依赖，且未调用GetAndResetSwaps()导致mapping一直增大
        // 删除原有写mapping语句：swapMapping_[srcBlockId] = dstBlockId;
        // 后续考虑swapMapping_的使用供后续扩展
        currentSwapMapping.push_back(std::make_pair(srcBlockId, dstBlockId));
    }
    return currentSwapMapping;
}

size_t CpuNpuBlockAllocator::GetNumFullBlocksTouched(const std::vector<BlockObjSPtr> &blocks,
                                                     DeviceType deviceType) const {
    if (rankSize_ == 1) {
        return GetAllocator(deviceType)->GetNumFullBlocksTouched(blocks);
    }

    size_t numFullBlocksTouched = 0;
    for (const BlockObjSPtr &block : blocks) {
        std::vector<BlockObjSPtr> eachBlock = {block};
        numFullBlocksTouched += GetAllocator(deviceType, block->GetRankIdx())->GetNumFullBlocksTouched(eachBlock);
    }
    return numFullBlocksTouched;
}

std::vector<std::pair<BlockId, BlockId>> CpuNpuBlockAllocator::ClearCopyOnWrites() {
    if (rankSize_ > 1) {
        throw std::runtime_error("ClearCopyOnWrites only supports single rank");
    }
    return GetAllocator(DeviceType::NPU)->ClearCopyOnWrites();
}

void CpuNpuBlockAllocator::MarkBlocksAsAccessed(size_t rankId, const std::vector<BlockId> &blockIds, float now) {
    if (blockIds.size() == 0) {
        return;
    }
    DeviceType deviceType = GetDeviceTypeForBlockId(blockIds[0]);
    // prefixcache sp cp场景下，只支持prefill阶段
    return GetAllocator(deviceType, rankId)->MarkBlocksAsAccessed(blockIds, now);
}

void CpuNpuBlockAllocator::MarkBlocksAsComputed() {
    // prefixcache sp cp场景下，只支持prefill阶段
    for (size_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        GetAllocator(DeviceType::NPU, rankIdx)->MarkBlocksAsComputed();
    }
    return;
}

size_t CpuNpuBlockAllocator::GetCachedBlockNum(size_t rankIdx, std::vector<HashValue> hashValues) const {
    size_t res = GetAllocator(DeviceType::NPU, rankIdx)->GetCachedBlockNum(hashValues);
    return res;
}

std::vector<size_t> CpuNpuBlockAllocator::GetAllRankCommonComputedBlockNum(
    const std::vector<std::vector<std::vector<BlockId>>> &rankedComputedSeqBlockIds) const {
    std::vector<size_t> allRankComputedBlockNum;
    for (size_t rankIdx = 0; rankIdx < rankedComputedSeqBlockIds.size(); rankIdx++) {
        std::vector<BlockId> computedBlocks =
            GetAllocator(DeviceType::NPU, rankIdx)->GetCommonComputedBlockIds(rankedComputedSeqBlockIds[rankIdx]);
        allRankComputedBlockNum.push_back(computedBlocks.size());
    }
    return allRankComputedBlockNum;
}

std::vector<BlockId> CpuNpuBlockAllocator::GetCommonComputedBlockIds(
    const std::vector<std::vector<BlockId>> &computedSeqBlockIds) const {
    // prefixcache sp cp场景下，只支持prefill阶段的，待实现
    return GetAllocator(DeviceType::NPU, 0)->GetCommonComputedBlockIds(computedSeqBlockIds);
}

float CpuNpuBlockAllocator::GetPrefixCacheHitRate() const {
    // prefixcache sp cp场景下，只支持prefill阶段的，待实现
    return GetAllocator(DeviceType::NPU, 0)->GetPrefixCacheHitRate();
}

bool CpuNpuBlockAllocator::ResetPrefixCache() const { return GetAllocator(DeviceType::NPU, 0)->ResetPrefixCache(); }

bool CpuNpuBlockAllocator::FindCachedBlockPrefix(size_t rankIdx, HashValue blockHash) const {
    // prefixcache sp cp场景下，只支持prefill阶段
    return GetAllocator(DeviceType::NPU, rankIdx)->FindCachedBlockPrefix(blockHash);
}

std::vector<BlockId> CpuNpuBlockAllocator::FindCachedBlocksPrefix(size_t rankIdx,
                                                                  std::vector<HashValue> &blockHashes) const {
    // prefixcache sp cp场景下，只支持prefill阶段
    return GetAllocator(DeviceType::NPU, rankIdx)->FindCachedBlocksPrefix(blockHashes);
}
// not used.
// usage scenario is required. If no one calling this, swaMapping_ will keep growing.
std::vector<std::pair<BlockId, BlockId>> CpuNpuBlockAllocator::GetAndResetSwaps() {
    std::vector<std::pair<BlockId, BlockId>> swapMapping = {};
    for (const auto &pair : swapMapping_) {
        swapMapping.push_back(pair);
    }
    swapMapping_.clear();
    return swapMapping;
}

void CpuNpuBlockAllocator::AppendTokenIds(BlockObjSPtr block, const std::vector<TokenId> &tokenIds) {
    BlockId blockId = block->GetBlockId();
    DeviceType deviceType = GetDeviceTypeForBlockId(blockId);
    GetAllocator(deviceType, block->GetRankIdx())->AppendTokenIds(block, tokenIds);
}

void CpuNpuBlockAllocator::ReplaceToken(BlockObjSPtr block, size_t startIndex, TokenId newToken) {
    BlockId blockId = block->GetBlockId();
    DeviceType deviceType = GetDeviceTypeForBlockId(blockId);
    GetAllocator(deviceType, block->GetRankIdx())->ReplaceToken(block, startIndex, newToken);
}

BlockAllocatorSPtr CpuNpuBlockAllocator::GetAllocator(DeviceType deviceType, size_t rankIdx) const {
    if (deviceType == DeviceType::CPU) {
        if (rankIdx >= cpuAllocators_.size()) {
            throw std::out_of_range("Invalid rank index for CPU allocator");
        }
        return cpuAllocators_[rankIdx];
    } else if (deviceType == DeviceType::NPU) {
        if (rankIdx >= npuAllocators_.size()) {
            throw std::out_of_range("Invalid rank index for NPU allocator");
        }
        return npuAllocators_[rankIdx];
    }
    throw std::invalid_argument("Unsupported device type");
}

}  // namespace mindie_llm
