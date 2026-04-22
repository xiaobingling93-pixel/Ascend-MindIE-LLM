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

#ifndef CPU_NPU_BLOCK_ALLOCATOR_H
#define CPU_NPU_BLOCK_ALLOCATOR_H

#include <unordered_map>

#include "device_aware_block_allocator.h"

/// 定义一个可以在CPU和NPU上分配KvCacheBlock的allocator.
///
/// 这个对象继承DeviceAwareBlockAllocator，并且提供在NPU和CPU上分配KvCacheBlock的接口
/// CpuNpuBlockAllocator分别维护了CPU和NPU的Block内存池
namespace mindie_llm {

class CpuNpuBlockAllocator : public DeviceAwareBlockAllocator {
   public:
    /// @brief
    /// @param allocatorType: HASHLESS or PREFIXCACHING. if caching is enabled, PREFIXCACHING should be used.
    /// @param numCpuBlocks: CPU Blocks.
    /// @param numNpuBlocks: NPU blocks.
    /// @param blockSize: block size, tokens num a block can contain.

    explicit CpuNpuBlockAllocator(const AllocatorConfig &config);

    // mutable block is the last block of a sequence which has empty slots and can be appended with tokens.
    BlockObjSPtr AllocateMutableBlock(DeviceType deviceType, std::vector<TokenId> &tokenIds,
                                      BlockObjSPtr prevBlock = nullptr, HashValue extraHash = 0,
                                      size_t rankIdx = 0) override;
    // immutable block is the prefix blocks of a sequence which is fully filled with tokens.
    BlockObjSPtr AllocateImmutableBlock(DeviceType deviceType, std::vector<TokenId> &tokenIds,
                                        BlockObjSPtr prevBlock = nullptr, HashValue extraHash = 0,
                                        size_t rankIdx = 0) override;

    std::vector<BlockObjSPtr> AllocateImmutableBlocks(DeviceType deviceType,
                                                      std::vector<std::vector<TokenId>> &tokenIds,
                                                      BlockObjSPtr prevBlock = nullptr, HashValue extraHash = 0,
                                                      size_t rankIdx = 0) override;
    // free block id owned by block, and release BlockObj into object pool. After the call, block should be empty and
    // can not be referenced.
    void Free(BlockObjSPtr &block) override;

    // fork a block list with shared block ids
    std::vector<BlockObjSPtr> Fork(BlockObjSPtr &block) override;

    size_t GetNumTotalBlocks(DeviceType deviceType) const override;

    size_t GetNumFreeBlock(DeviceType deviceType) const override;

    size_t GetNumFreeBlock(DeviceType deviceType, size_t rankId) const override;

    // assume npu block id is 0~99, and cpu block id is 100~199, block 101 has physical block id 1. Physical block id is
    // used to calculate address offset in executor.
    PhysicalBlockId GetPhysicalBlockId(BlockId globalBlockId) const override;

    // swap srcBlocks out of srcDevice into dstDevice and return the src block id --> dst block id mapping.
    std::vector<std::pair<BlockId, BlockId>> Swap(std::vector<BlockObjSPtr> &swapTargetBlocks, DeviceType srcDevice,
                                                  DeviceType dstDevice) override;
    // the blocks num is full in this scheduling round , and will be considered computed in the beginning of next round.
    size_t GetNumFullBlocksTouched(const std::vector<BlockObjSPtr> &blocks, DeviceType deviceType) const override;
    // cow is actually not used with current prefix cache setup.
    std::vector<std::pair<BlockId, BlockId>> ClearCopyOnWrites() override;

    void MarkBlocksAsAccessed(size_t rankId, const std::vector<BlockId> &blockIds, float now) override;
    // only touched blocks will be marked as computed. and the blockIds parameter is not used.
    void MarkBlocksAsComputed() override;
    // Usage scenario required.
    std::vector<BlockId> GetCommonComputedBlockIds(
        const std::vector<std::vector<BlockId>> &computedSeqBlockIds) const override;

    std::vector<size_t> GetAllRankCommonComputedBlockNum(
        const std::vector<std::vector<std::vector<BlockId>>> &rankedComputedSeqBlockIds) const override;

    size_t GetCachedBlockNum(size_t rankIdx, std::vector<HashValue> hashValues) const override;

    float GetPrefixCacheHitRate() const override;

    bool ResetPrefixCache() const override;

    bool FindCachedBlockPrefix(size_t rankIdx, HashValue blockHash) const override;

    // longest computed and cached blocks prefix of given rankIdx and block hashes list
    std::vector<BlockId> FindCachedBlocksPrefix(size_t rankIdx, std::vector<HashValue> &blockHashes) const override;

    std::vector<std::pair<BlockId, BlockId>> GetAndResetSwaps();

    void AppendTokenIds(BlockObjSPtr block, const std::vector<TokenId> &tokenIds) override;

    void ReplaceToken(BlockObjSPtr block, size_t startIndex, TokenId newToken) override;

   private:
    size_t rankSize_;

    std::vector<BlockAllocatorSPtr> cpuAllocators_;  // CPU有rankSize个allocator

    std::vector<BlockAllocatorSPtr> npuAllocators_;  // NPU有rankSize个allocator

    std::unordered_map<BlockId, BlockId> swapMapping_;

    // +++由于删除了blockid和allocator的查找表，加一个npu和cpu blockid的分界值，区分某一blockid是何种block
    BlockId beginCpuBlockId_;

    size_t hostSize_;

    DeviceType GetDeviceTypeForBlockId(BlockId blockId) const;

    BlockAllocatorSPtr GetAllocator(DeviceType deviceType, size_t rankIdx = 0) const;
};
}  // namespace mindie_llm

#endif
