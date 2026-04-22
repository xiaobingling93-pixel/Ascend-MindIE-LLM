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

#include <algorithm>
#include <iostream>
#include <memory>

#include "hashless_block_obj.h"

namespace mindie_llm {

struct AllocatorConfig {
    BlockAllocatorType allocatorType;
    size_t numCpuBlocks;
    size_t numNpuBlocks;
    size_t blockSize;
    size_t rankSize = 1;
    size_t hostSize = 1;
};

// 支持多种设备的内存分配器
class DeviceAwareBlockAllocator {
   public:
    DeviceAwareBlockAllocator() = default;

    virtual ~DeviceAwareBlockAllocator() = default;

    virtual BlockObjSPtr AllocateMutableBlock(DeviceType deviceType, std::vector<TokenId> &tokenIds,
                                              BlockObjSPtr prevBlock = nullptr, HashValue extraHash = 0,
                                              size_t rankIdx = 0) = 0;

    virtual BlockObjSPtr AllocateImmutableBlock(DeviceType deviceType, std::vector<TokenId> &tokenIds,
                                                BlockObjSPtr prevBlock = nullptr, HashValue extraHash = 0,
                                                size_t rankIdx = 0) = 0;

    virtual std::vector<BlockObjSPtr> AllocateImmutableBlocks(DeviceType deviceType,
                                                              std::vector<std::vector<TokenId>> &tokenIds,
                                                              BlockObjSPtr prevBlock = nullptr, HashValue extraHash = 0,
                                                              size_t rankIdx = 0) = 0;

    virtual size_t GetNumFreeBlock(DeviceType blockType) const = 0;

    virtual size_t GetNumFreeBlock(DeviceType blockType, size_t rankId) const = 0;

    virtual size_t GetNumTotalBlocks(DeviceType blockType) const = 0;

    virtual void Free(BlockObjSPtr &block) = 0;  // 会根据blockid找到device

    virtual std::vector<BlockObjSPtr> Fork(BlockObjSPtr &lastBlock) = 0;

    virtual std::vector<std::pair<BlockId, BlockId>> ClearCopyOnWrites() = 0;

    virtual void MarkBlocksAsAccessed(size_t rankId, const std::vector<BlockId> &blockIds, float now) = 0;

    virtual void MarkBlocksAsComputed() = 0;

    virtual std::vector<BlockId> GetCommonComputedBlockIds(
        const std::vector<std::vector<BlockId>> &computedSeqBlockIds) const = 0;

    virtual std::vector<size_t> GetAllRankCommonComputedBlockNum(
        const std::vector<std::vector<std::vector<BlockId>>> &rankedComputedSeqBlockIds) const = 0;

    virtual size_t GetCachedBlockNum(size_t rankIdx, std::vector<HashValue> hashValues) const = 0;

    virtual size_t GetNumFullBlocksTouched(const std::vector<BlockObjSPtr> &blocks, const DeviceType device) const = 0;

    virtual std::vector<std::pair<BlockId, BlockId>> Swap(std::vector<BlockObjSPtr> &srcBlocks, DeviceType srcDevice,
                                                          DeviceType dstDevice) = 0;

    virtual PhysicalBlockId GetPhysicalBlockId(BlockId globalBlockId) const = 0;

    virtual float GetPrefixCacheHitRate() const = 0;

    virtual bool ResetPrefixCache() const = 0;

    virtual bool FindCachedBlockPrefix(size_t rankIdx, HashValue blockHash) const = 0;

    // 找到指定rank上的最长匹配的前缀的blockid lists
    virtual std::vector<BlockId> FindCachedBlocksPrefix(size_t rankIdx, std::vector<HashValue> &blockHashes) const = 0;

    virtual void AppendTokenIds(BlockObjSPtr blockObj, const std::vector<TokenId> &tokenIds) = 0;

    virtual void ReplaceToken(BlockObjSPtr blockObj, size_t startIndex, TokenId newToken) = 0;
};

using DeviceAwareBlockAllocatorSPtr = std::shared_ptr<DeviceAwareBlockAllocator>;
}  // namespace mindie_llm
