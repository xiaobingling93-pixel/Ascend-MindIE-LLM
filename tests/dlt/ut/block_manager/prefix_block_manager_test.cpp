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
 
#include <gtest/gtest.h>
#include <cmath>
#include <unordered_map>
#include <cassert>
#define private public
#include "self_attn_block_manager.h"
#include "sequence_group.h"
#include "prefix_cache_block_allocator.h"
#include "ref_counter.h"

namespace mindie_llm {
struct AllocateTestData {
    SequenceId seqId;
    std::vector<TokenId> inputs;
    RequestId requestId;
    size_t blockSize;
    size_t cpuBlockNum;
    size_t npuBlockNum;
    bool enableCaching;
    AllocStatus canAllocate;
};

struct ConsistencyTestData {
    SequenceId seqId;
    std::vector<TokenId> inputs;
    std::vector<TokenId> tokensToAppend;
    std::vector<TokenId> moreTokensToAppend;
    RequestId requestId;
    size_t blockSize;
    size_t cpuBlockNum;
    size_t npuBlockNum;
    bool enableCaching;
};

struct SwapOutTestData {
    SequenceId seqId;
    std::vector<TokenId> inputs;
    RequestId requestId;
    size_t blockSize;
    size_t cpuBlockNum;
    size_t npuBlockNum;
    bool enableCaching;
    AllocStatus canAllocate;
    bool canSwapOut;
    std::vector<std::pair<PhysicalBlockId, PhysicalBlockId>> physicalBlockIdMappingExpected;
    std::vector<BlockId> blockIdsExpected;
};
RefCount CalRefCnts(RefCounterProtocol *refCounter)
{
    RefCount refCntSum = 0;
    std::unordered_map<BlockId, RefCount> refCountsMap = static_cast<RefCounter *>(refCounter)->refCounts_;
    for (const auto pair : refCountsMap) {
        refCntSum += pair.second;
    }
    return refCntSum;
};
void SwapOutTestHelper(const std::vector<AllocateTestData> &sequencialAllocateTestData,
                       const std::vector<SwapOutTestData> &sequencialSwapOutTestData)
{
    size_t blockSize = sequencialAllocateTestData[0].blockSize;
    size_t cpuBlockNum = sequencialAllocateTestData[0].cpuBlockNum;
    size_t npuBlockNum = sequencialAllocateTestData[0].npuBlockNum;
    bool enableCaching = sequencialAllocateTestData[0].enableCaching;
    size_t reservedBlockNum = 0;
    size_t speculativeSlots = 0;
    std::shared_ptr<SamplingParams> sampling = nullptr;
    // 初始化测试数据
    BlockManagerConfig config{blockSize, cpuBlockNum, npuBlockNum, reservedBlockNum, speculativeSlots, enableCaching};
    SelfAttnBlockManager blockManager(config);

    std::vector<SequenceGroupSPtr> seqGroupPtrs;
    std::vector<SequenceSPtr> seqPtrs;
    int extractFactor = 4;
    auto cpuNpuAllocator = static_cast<CpuNpuBlockAllocator *>(blockManager.blockAllocator_.get());
    auto npuPrefixAllocator = cpuNpuAllocator->GetAllocator(DeviceType::NPU);
    auto cpuPrefixAllocator = cpuNpuAllocator->GetAllocator(DeviceType::CPU);
    auto blockPool = static_cast<PrefixCacheBlockAllocator *>(npuPrefixAllocator.get())->blockObjPool_.get();

    RefCounterProtocol *npuRefCounter =
        static_cast<PrefixCacheBlockAllocator *>(npuPrefixAllocator.get())->refCounter_.get();
    RefCounterProtocol *cpuRefCounter =
        static_cast<PrefixCacheBlockAllocator *>(cpuPrefixAllocator.get())->refCounter_.get();

    std::vector<int> AllocatedNumBlocks;

    int currNumFreeBlocksObjs = blockPool->GetPoolSize();
    int currNumFreeCpuBlockIds = cpuBlockNum;
    int currNumFreeNpuBlockIds = npuBlockNum;

    std::unordered_map<HashValue, int> npuBlockHashOccurences; //记录npu上block的引用次数
    std::unordered_map<HashValue, int> cpuBlockHashOccurences; //记录cpu上block的引用次数
    for (auto &allocateTestData : sequencialAllocateTestData) {

        SequenceSPtr seqPtr = std::make_shared<Sequence>(allocateTestData.seqId, allocateTestData.blockSize,
                                                         allocateTestData.inputs);
        std::vector<std::shared_ptr<Sequence>> seqs = {seqPtr};
        RequestId thisRequestId = allocateTestData.requestId;
        SequenceGroupSPtr groupPtr = std::make_shared<SequenceGroup>(thisRequestId, seqs, sampling);
        seqGroupPtrs.push_back(groupPtr);
        seqPtrs.push_back(seqPtr);

        EXPECT_EQ(blockPool->GetFreeObjNum(), currNumFreeBlocksObjs);

        AllocStatus canAllocate = blockManager.CanAllocate(groupPtr);
        EXPECT_EQ(canAllocate, allocateTestData.canAllocate); // CHECK: CanAllocate

        blockManager.Allocate(groupPtr); // CHECK: Allocate
        //手动模拟allocate
        int numAllocatedBlocks = std::ceil(allocateTestData.inputs.size() * 1.0 / allocateTestData.blockSize);
        AllocatedNumBlocks.push_back(numAllocatedBlocks);

        currNumFreeBlocksObjs -= numAllocatedBlocks;
        currNumFreeNpuBlockIds -= numAllocatedBlocks;
        auto blocks = blockManager.seqId2BlockTable_.at(seqPtr->seqId_).GetBlockObjs();
        for (auto block : blocks) {
            HashValue blockHash = block->PrefixHash();
            if (blockHash == INVALID_HASH_VALUE) { // not full
                continue;
            }
            if (npuBlockHashOccurences.find(blockHash) != npuBlockHashOccurences.end() &&
                npuBlockHashOccurences[blockHash] > 0) {
                currNumFreeNpuBlockIds++; //可以被重用的block，不应该占用blockid
                npuBlockHashOccurences[blockHash]++;
            } else
                npuBlockHashOccurences[blockHash] = 1;
        }
        EXPECT_EQ(extractFactor * (cpuBlockNum + npuBlockNum) - currNumFreeBlocksObjs,
                  CalRefCnts(cpuRefCounter) + CalRefCnts(npuRefCounter)); // CHECK: ref count
        EXPECT_EQ(blockPool->GetFreeObjNum(), currNumFreeBlocksObjs);
        EXPECT_EQ(blockManager.GetNumFreeNpuBlocks(), currNumFreeNpuBlockIds);
    }

    int i = 0;
    for (const auto &test : sequencialSwapOutTestData) {

        auto seqPtr = seqPtrs[i];
        auto groupPtr = seqGroupPtrs[i];
        // 设置序列状态为 RUNNING
        seqPtr->status_ = SequenceStatus::RUNNING;

        EXPECT_EQ(blockPool->GetFreeObjNum(), currNumFreeBlocksObjs); //没fix bug，检查blockobj数量结果不对

        bool canSwapOut = blockManager.CanSwapOut(groupPtr);
        EXPECT_EQ(canSwapOut, test.canSwapOut); // 检查 CanSwapOut 是否符合预期

        EXPECT_EQ(blockPool->GetFreeObjNum(), currNumFreeBlocksObjs); //没fix bug，检查blockobj数量结果不对

        //手动模拟swapout
        currNumFreeCpuBlockIds -= AllocatedNumBlocks[i];
        currNumFreeNpuBlockIds += AllocatedNumBlocks[i];
        auto blocks = blockManager.seqId2BlockTable_.at(seqPtr->seqId_).GetBlockObjs();
        for (auto block : blocks) {
            HashValue blockHash = block->PrefixHash();
            if (blockHash == INVALID_HASH_VALUE) {
                continue;
            }
            assert(npuBlockHashOccurences.find(blockHash) != npuBlockHashOccurences.end()); // 这个block应该在npu上
            assert(npuRefCounter->GetRefCount(block->GetBlockId()) ==
                   npuBlockHashOccurences[blockHash]); //经过allocate之后应该refcount正确
            npuBlockHashOccurences[blockHash]--;       //模拟swapout
            if (npuBlockHashOccurences[blockHash] != 0) {
                currNumFreeNpuBlockIds--; //只有当blockHash的引用计数为0时，才会free blockid
            }
            if (cpuBlockHashOccurences.find(blockHash) != cpuBlockHashOccurences.end() &&
                cpuBlockHashOccurences[blockHash] > 0) {
                currNumFreeCpuBlockIds++; //可以被重用的block，不应该占用blockid
                cpuBlockHashOccurences[blockHash]++;
            } else
                cpuBlockHashOccurences[blockHash] = 1;
        }

        auto physicalBlockIdMapping = blockManager.SwapOut(groupPtr);

        EXPECT_EQ(physicalBlockIdMapping, test.physicalBlockIdMappingExpected); // 检查物理块映射是否符合预期

        for (auto block : blocks) {
            HashValue blockHash = block->PrefixHash();
            if (blockHash == INVALID_HASH_VALUE) {
                continue;
            }
            assert(cpuRefCounter->GetRefCount(block->GetBlockId()) == cpuBlockHashOccurences[blockHash]);
        }
        EXPECT_EQ(blockPool->GetFreeObjNum(), currNumFreeBlocksObjs); //没fix bug，检查blockobj数量结果不对
        EXPECT_EQ(extractFactor * (cpuBlockNum + npuBlockNum) - currNumFreeBlocksObjs,
                  CalRefCnts(cpuRefCounter) + CalRefCnts(npuRefCounter)); // CHECK: ref count
        EXPECT_EQ(blockManager.GetNumFreeCpuBlocks(), currNumFreeCpuBlockIds);
        EXPECT_EQ(blockManager.GetNumFreeNpuBlocks(), currNumFreeNpuBlockIds);

        auto blockIds = blockManager.GetBlockIds(seqPtr->seqId_);
        ASSERT_EQ(blockIds.size(), 1u);
        EXPECT_EQ(blockIds[0], test.blockIdsExpected); // 检查块 ID 是否符合预期
        i++;
    }
}

/**
 * Swapout测试场景1：sequence的token都占满整个block，整个sequence都能击中kv缓存。allocate满整个NPU blocks+全部swapout
 * 预期结果：
 * 1. allocate+swapout成功，最后blocks都在CPU上，对应的block数量、blockID正确、swapout返回的mapping正确
 */

TEST(PrefixSwapOutTest, ExpectSuccessHavingMultipleAllocateAndSwapOutWhenFullyCache)
{
    const std::vector<AllocateTestData> sequencialAllocateTestData = {
        AllocateTestData{10, {1, 2, 3, 4, 5, 6, 7, 8}, "Req:0", 8, 2, 2, true, AllocStatus::OK},
        AllocateTestData{21, {1, 2, 3, 4, 5, 6, 7, 8}, "Req:1", 8, 2, 2, true, AllocStatus::OK},
        AllocateTestData{36, {21, 22, 23, 24, 25, 26}, "Req:2", 8, 2, 2, true, AllocStatus::OK},
    };
    const std::vector<SwapOutTestData> sequencialSwapOutTestData = {
        //在这个测试场景，只有SwapOutTestData最后3个字段需要比较
        SwapOutTestData{-1, {}, "", 0, 0, 0, false, AllocStatus::OK, true, {{0, 0}}, {2}},
        SwapOutTestData{-1, {}, "", 0, 0, 0, false, AllocStatus::OK, true, {{0, 0}}, {2}},
        // swapout前后block都指向同一个物理块2，5表明块在CPU上
        SwapOutTestData{-1, {}, "", 0, 0, 0, false, AllocStatus::OK, true, {{1, 1}}, {3}},
    };
    SwapOutTestHelper(sequencialAllocateTestData, sequencialSwapOutTestData);
}

/**
 * Swapout测试场景2：sequence前缀全部匹配能击中kv缓存，前缀部分匹配不算击中kv缓存。allocate满整个NPU
 blocks+全部swapout
 * 预期结果： 1.allocate+swapout成功，最后blocks都在CPU上，对应的blockID数量正确、blockobj数量正确、swapout返回的mapping正确，refcount正确
 */

TEST(PrefixSwapOutTest, ExpectSuccessHavingMultipleAllocateAndSwapOutWhenPartialCache)
{
    const std::vector<AllocateTestData> sequencialAllocateTestData = {
        AllocateTestData{
            10, {1, 2, 3, 4, 5, 6, 7, 8, 21, 22, 23, 24, 25, 26, 27, 28}, "Req:0", 8, 5, 5, true, AllocStatus::OK},
        AllocateTestData{21, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, "Req:1", 8, 5, 5, true, AllocStatus::OK},
        AllocateTestData{36, {21, 22, 23, 24, 25, 26, 27, 28, 29}, "Req:2", 8, 5, 5, true, AllocStatus::OK},
    };
    const std::vector<SwapOutTestData> sequencialSwapOutTestData = {
        //在这个测试场景，只有SwapOutTestData最后3个字段需要比较
        SwapOutTestData{-1, {}, "", 0, 0, 0, false, AllocStatus::OK, true, {{0, 0}, {1, 1}}, {5, 6}},
        SwapOutTestData{-1, {}, "", 0, 0, 0, false, AllocStatus::OK, true, {{0, 0}, {2, 2}}, {5, 7}},
        // swapout前后block都指向同一个物理块3，9表明块在CPU上
        SwapOutTestData{-1, {}, "", 0, 0, 0, false, AllocStatus::OK, true, {{3, 3}, {4, 4}}, {8, 9}},
    };
    SwapOutTestHelper(sequencialAllocateTestData, sequencialSwapOutTestData);
}

/**
 * Swapout测试场景3：sequence前缀全部匹配能击中kv缓存。allocate满整个NPU blocks+全部swapout
 * 预期结果： 1.allocate+swapout成功，最后blocks都在CPU上，对应的blockID数量正确、blockobj数量正确、swapout返回的mapping正确，refcount正确
 */

TEST(PrefixSwapOutTest, ExpectSuccessHavingMultipleAllocateAndSwapOutWhenPartialCache2)
{
    const std::vector<AllocateTestData> sequencialAllocateTestData = {
        AllocateTestData{
            10, {1, 2, 3, 4, 5, 6, 7, 8, 21, 22, 23, 24, 25, 26, 27, 28}, "Req:0", 8, 6, 6, true, AllocStatus::OK},
        AllocateTestData{21, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, "Req:1", 8, 4, 4, true, AllocStatus::OK},
        AllocateTestData{
            36, {1, 2, 3, 4, 5, 6, 7, 8, 21, 22, 23, 24, 25, 26, 27, 28, 29}, "Req:2", 8, 6, 6, true, AllocStatus::OK},
    };
    const std::vector<SwapOutTestData> sequencialSwapOutTestData = {
        //在这个测试场景，只有SwapOutTestData最后3个字段需要比较
        SwapOutTestData{-1, {}, "", 0, 0, 0, false, AllocStatus::OK, true, {{0, 0}, {1, 1}}, {6, 7}},
        SwapOutTestData{-1, {}, "", 0, 0, 0, false, AllocStatus::OK, true, {{0, 0}, {2, 2}}, {6, 8}},
        // 第0，1块都可以复用，第2块使用新的blockid，换出到cpu上也可以复用
        SwapOutTestData{-1, {}, "", 0, 0, 0, false, AllocStatus::OK, true, {{0, 0}, {1, 1}, {3, 3}}, {6, 7, 9}},
    };
    SwapOutTestHelper(sequencialAllocateTestData, sequencialSwapOutTestData);
}

struct SwapInTestData {
    SequenceId seqId;
    std::vector<TokenId> inputs;
    RequestId requestId;
    size_t blockSize;
    size_t cpuBlockNum;
    size_t npuBlockNum;
    bool enableCaching;
    AllocStatus canAllocate;
    AllocStatus canSwapIn;
    std::vector<std::pair<PhysicalBlockId, PhysicalBlockId>> physicalBlockIdMappingExpected;
    std::vector<BlockId> blockIdsExpected;
};

void SwapInTestHelper(const std::vector<AllocateTestData> &sequencialAllocateTestData,
                      const std::vector<SwapInTestData> &sequencialSwapInTestData)
{
    size_t blockSize = sequencialAllocateTestData[0].blockSize;
    size_t cpuBlockNum = sequencialAllocateTestData[0].cpuBlockNum;
    size_t npuBlockNum = sequencialAllocateTestData[0].npuBlockNum;
    bool enableCaching = sequencialAllocateTestData[0].enableCaching;
    size_t reservedBlockNum = 0;
    size_t speculativeSlots = 0;
    std::shared_ptr<SamplingParams> sampling = nullptr;
    // 初始化测试数据
    BlockManagerConfig config{blockSize, cpuBlockNum, npuBlockNum, reservedBlockNum, speculativeSlots, enableCaching};
    SelfAttnBlockManager blockManager(config);

    std::vector<SequenceGroupSPtr> seqGroupPtrs;
    std::vector<SequenceSPtr> seqPtrs;
    int extractFactor = 4;
    auto cpuNpuAllocator = static_cast<CpuNpuBlockAllocator *>(blockManager.blockAllocator_.get());
    auto npuPrefixAllocator = cpuNpuAllocator->GetAllocator(DeviceType::NPU);
    auto cpuPrefixAllocator = cpuNpuAllocator->GetAllocator(DeviceType::CPU);
    auto blockPool = static_cast<PrefixCacheBlockAllocator *>(npuPrefixAllocator.get())->blockObjPool_.get();

    auto npuRefCounter = static_cast<PrefixCacheBlockAllocator *>(npuPrefixAllocator.get())->refCounter_.get();
    auto cpuRefCounter = static_cast<PrefixCacheBlockAllocator *>(cpuPrefixAllocator.get())->refCounter_.get();

    std::vector<int> AllocatedNumBlocks;

    int currNumFreeBlocksObjs = blockPool->GetPoolSize();

    int currNumFreeCpuBlockIds = cpuBlockNum;
    int currNumFreeNpuBlockIds = npuBlockNum;
    int numTestData = sequencialSwapInTestData.size();

    std::unordered_map<HashValue, int> npuBlockHashOccurences; //记录npu上block的引用次数
    std::unordered_map<HashValue, int> cpuBlockHashOccurences; //记录cpu上block的引用次数
    for (auto &allocateTestData : sequencialAllocateTestData) {

        SequenceSPtr seqPtr = std::make_shared<Sequence>(allocateTestData.seqId, allocateTestData.blockSize,
                                                         allocateTestData.inputs);
        std::vector<std::shared_ptr<Sequence>> seqs = {seqPtr};
        RequestId thisRequestId = allocateTestData.requestId;
        SequenceGroupSPtr groupPtr = std::make_shared<SequenceGroup>(thisRequestId, seqs, sampling);
        seqGroupPtrs.push_back(groupPtr);
        seqPtrs.push_back(seqPtr);

        AllocStatus canAllocate = blockManager.CanAllocate(groupPtr);

        bool res = blockManager.Allocate(groupPtr); // CHECK: Allocate
        //手动模拟allocate
        int numAllocatedBlocks = std::ceil(allocateTestData.inputs.size() * 1.0 / allocateTestData.blockSize);
        AllocatedNumBlocks.push_back(numAllocatedBlocks);

        currNumFreeBlocksObjs -= numAllocatedBlocks;
        currNumFreeNpuBlockIds -= numAllocatedBlocks;
        auto blocks = blockManager.seqId2BlockTable_.at(seqPtr->seqId_).GetBlockObjs();
        for (auto block : blocks) {
            HashValue blockHash = block->PrefixHash();
            if (blockHash == INVALID_HASH_VALUE) { // not full
                continue;
            }
            if (npuBlockHashOccurences.find(blockHash) != npuBlockHashOccurences.end() &&
                npuBlockHashOccurences[blockHash] > 0) {
                currNumFreeNpuBlockIds++; //可以被重用的block，不应该占用blockid
                npuBlockHashOccurences[blockHash]++;
            } else
                npuBlockHashOccurences[blockHash] = 1;
        }
    }
    // swapout
    for (int i = 0; i < numTestData; i++) {
        const auto test = sequencialSwapInTestData[i];
        auto seqPtr = seqPtrs[i];
        auto groupPtr = seqGroupPtrs[i];
        // 设置序列状态为 RUNNING
        seqPtr->status_ = SequenceStatus::RUNNING;

        bool canSwapOut = blockManager.CanSwapOut(groupPtr);

        //手动模拟swapout
        currNumFreeCpuBlockIds -= AllocatedNumBlocks[i];
        currNumFreeNpuBlockIds += AllocatedNumBlocks[i];
        auto blocks = blockManager.seqId2BlockTable_.at(seqPtr->seqId_).GetBlockObjs();
        for (auto block : blocks) {
            HashValue blockHash = block->PrefixHash();
            if (blockHash == INVALID_HASH_VALUE) {
                continue;
            }
            npuBlockHashOccurences[blockHash]--;
            if (npuBlockHashOccurences[blockHash] != 0) {
                currNumFreeNpuBlockIds--; //只有当blockHash的引用计数为0时，才会free blockid
            }
            if (cpuBlockHashOccurences.find(blockHash) != cpuBlockHashOccurences.end() &&
                cpuBlockHashOccurences[blockHash] > 0) {
                currNumFreeCpuBlockIds++; //可以被重用的block，不应该占用blockid
                cpuBlockHashOccurences[blockHash]++;
            } else
                cpuBlockHashOccurences[blockHash] = 1;
        }
        auto physicalBlockIdMapping = blockManager.SwapOut(groupPtr);
    }

    for (int i = 0; i < numTestData; i++) {
        auto test = sequencialSwapInTestData[i];
        auto seqPtr = seqPtrs[i];
        auto groupPtr = seqGroupPtrs[i];
        // 设置序列状态为 SWAPPED
        seqPtr->status_ = SequenceStatus::SWAPPED;

        EXPECT_EQ(blockPool->GetFreeObjNum(), currNumFreeBlocksObjs); //没fix bug，检查blockobj数量结果不对

        AllocStatus canSwapIn = blockManager.CanSwapIn(groupPtr, 0);
        EXPECT_EQ(canSwapIn, test.canSwapIn);

        //手动模拟swapin
        currNumFreeCpuBlockIds += AllocatedNumBlocks[i];
        currNumFreeNpuBlockIds -= AllocatedNumBlocks[i];
        auto blocks = blockManager.seqId2BlockTable_.at(seqPtr->seqId_).GetBlockObjs();
        for (auto block : blocks) {
            HashValue blockHash = block->PrefixHash();
            if (blockHash == INVALID_HASH_VALUE) {
                continue;
            }
            assert(cpuBlockHashOccurences.find(blockHash) != cpuBlockHashOccurences.end()); // 这个block应该在cpu上
            assert(cpuRefCounter->GetRefCount(block->GetBlockId()) ==
                   cpuBlockHashOccurences[blockHash]); //经过allocate之后应该refcount正确
            cpuBlockHashOccurences[blockHash]--;       //模拟swapin
            if (cpuBlockHashOccurences[blockHash] != 0) {
                currNumFreeCpuBlockIds--; //只有当blockHash的引用计数为0时，才会free blockid
            }
            if (npuBlockHashOccurences.find(blockHash) != npuBlockHashOccurences.end() &&
                npuBlockHashOccurences[blockHash] > 0) {
                currNumFreeNpuBlockIds++; //可以被重用的block，不应该占用blockid
                npuBlockHashOccurences[blockHash]++;
            } else
                npuBlockHashOccurences[blockHash] = 1;
        }

        auto physicalBlockIdMapping = blockManager.SwapIn(groupPtr);

        EXPECT_EQ(physicalBlockIdMapping, test.physicalBlockIdMappingExpected); // 检查物理块映射是否符合预期

        for (auto block : blocks) {
            HashValue blockHash = block->PrefixHash();
            if (blockHash == INVALID_HASH_VALUE) {
                continue;
            }
            assert(npuRefCounter->GetRefCount(block->GetBlockId()) == npuBlockHashOccurences[blockHash]);
        }
        EXPECT_EQ(blockPool->GetFreeObjNum(), currNumFreeBlocksObjs); //没fix bug，检查blockobj数量结果不对
        EXPECT_EQ(extractFactor * (cpuBlockNum + npuBlockNum) - currNumFreeBlocksObjs,
                  CalRefCnts(cpuRefCounter) + CalRefCnts(npuRefCounter)); // CHECK: ref count
        EXPECT_EQ(blockManager.GetNumFreeCpuBlocks(), currNumFreeCpuBlockIds);
        EXPECT_EQ(blockManager.GetNumFreeNpuBlocks(), currNumFreeNpuBlockIds);

        auto blockIds = blockManager.GetBlockIds(seqPtr->seqId_);
        ASSERT_EQ(blockIds.size(), 1u);
        EXPECT_EQ(blockIds[0], test.blockIdsExpected); // 检查块 ID 是否符合预期

        seqPtr->status_ = SequenceStatus::RUNNING;
        blockManager.GetCommonComputedBlockIds(groupPtr->GetSequences(SequenceStatus::RUNNING));
        blockManager.Free(seqPtr->seqId_); //如果block是满的，加入evitor，但是blockid不释放

        currNumFreeNpuBlockIds += AllocatedNumBlocks[i]; //这里满的block会被加入到evictor中，不满的直接释放
        currNumFreeBlocksObjs += AllocatedNumBlocks[i];
        for (auto block : blocks) {
            HashValue blockHash = block->PrefixHash();
            if (blockHash == INVALID_HASH_VALUE) { // not full
                continue;
            }
            npuBlockHashOccurences[blockHash]--;
            assert(npuBlockHashOccurences[blockHash] == 0);
            EXPECT_THROW(npuRefCounter->GetRefCount(block->GetBlockId()), std::runtime_error); // refcount=0了
        }
        EXPECT_EQ(extractFactor * (cpuBlockNum + npuBlockNum) - currNumFreeBlocksObjs,
                  CalRefCnts(cpuRefCounter) + CalRefCnts(npuRefCounter)); // CHECK: ref count
        EXPECT_EQ(blockManager.GetNumFreeNpuBlocks(), currNumFreeNpuBlockIds);
    }
}

/**
 * Swapin测试场景1：序列tokenid不存在重用，全部是满的block。全部allocate+全部swapout+全部swapin+free。
 * 预期结果：
 * 1. 流程能跑通，最后blocks都释放了，对应的blockID数量正确、blockobj数量正确、swapout返回的mapping正确，refcount正确
 */

TEST(PrefixSwapInTest, NoReuseAndFullBlock)
{
    const std::vector<AllocateTestData> sequencialAllocateTestData = {
        AllocateTestData{101, {11, 2, 3, 4, 5, 6, 7, 8}, "Req:0", 8, 2, 2, true, AllocStatus::OK},
        AllocateTestData{21, {11, 12, 13, 14, 15, 16, 17, 18}, "Req:1", 8, 2, 2, true, AllocStatus::OK},
    };
    const std::vector<SwapInTestData> sequencialSwapInTestData = {
        //在这个测试场景，只有SwapOutTestData最后3个字段需要比较
        SwapInTestData{-1, {}, "", 0, 0, 0, false, AllocStatus::OK, AllocStatus::OK, {{0, 0}}, {0}},
        SwapInTestData{-1, {}, "", 0, 0, 0, false, AllocStatus::OK, AllocStatus::OK, {{1, 1}}, {1}},
    };
    SwapInTestHelper(sequencialAllocateTestData, sequencialSwapInTestData);
}

/**
 * Swapin测试场景2：序列tokenid不存在重用，有不满的block。全部allocate+全部swapout+全部swapin+free。
 */
TEST(PrefixSwapInTest, NoReuseAndFullBlockPlusNonFullBlock)
{
    const std::vector<AllocateTestData> sequencialAllocateTestData = {
        AllocateTestData{101, {11, 2, 3, 4, 5, 6, 7, 8, 9, 10}, "Req:0", 8, 4, 4, true, AllocStatus::OK},
        AllocateTestData{21, {11, 12, 13, 14, 15, 16, 17, 18, 9, 10}, "Req:1", 8, 4, 4, true, AllocStatus::OK},
    };
    const std::vector<SwapInTestData> sequencialSwapInTestData = {
        //在这个测试场景，只有SwapOutTestData最后3个字段需要比较
        SwapInTestData{-1, {}, "", 0, 0, 0, false, AllocStatus::OK, AllocStatus::OK, {{0, 0}, {1, 1}}, {0, 1}},
        SwapInTestData{-1, {}, "", 0, 0, 0, false, AllocStatus::OK, AllocStatus::OK, {{2, 2}, {3, 3}}, {2, 3}},
    };
    SwapInTestHelper(sequencialAllocateTestData, sequencialSwapInTestData);
}

/**
 * Swapin测试场景3：序列tokenid存在重用，有不满的block。全部allocate+全部swapout+全部swapin+free。
 */
TEST(PrefixSwapInTest, WithReuseAndFullBlockPlusNonFullBlock)
{
    const std::vector<AllocateTestData> sequencialAllocateTestData = {
        AllocateTestData{1, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, "Req:0", 8, 8, 8, true, AllocStatus::OK},
        AllocateTestData{2, {11, 12, 13, 14, 15, 16, 17, 18, 22, 23}, "Req:1", 8, 8, 8, true, AllocStatus::OK},
        AllocateTestData{3, {1, 2, 3, 4, 5, 6, 7, 8, 22, 23}, "Req:3", 8, 8, 8, true, AllocStatus::OK},
        AllocateTestData{4, {11, 12, 13, 14, 15, 16, 17, 18, 88, 99}, "Req:4", 8, 8, 8, true, AllocStatus::OK},

    };
    const std::vector<SwapInTestData> sequencialSwapInTestData = {
        //在这个测试场景，只有SwapOutTestData最后3个字段需要比较
        //{0，0}的0是因为swapin的时候对于tokenid为{1, 2, 3, 4, 5, 6, 7, 8}的block 能找到cache，重用id；
        //{1, 6}的6是因为对于没满的tokenid为{9,10}的block，swapin的时候会分配新的blockid
        SwapInTestData{-1, {}, "", 0, 0, 0, false, AllocStatus::OK, AllocStatus::OK, {{0, 0}, {1, 6}}, {0, 6}},
        SwapInTestData{-1, {}, "", 0, 0, 0, false, AllocStatus::OK, AllocStatus::OK, {{2, 2}, {3, 7}}, {2, 7}},
        SwapInTestData{-1, {}, "", 0, 0, 0, false, AllocStatus::OK, AllocStatus::OK, {{0, 0}, {4, 1}}, {0, 1}},
        SwapInTestData{-1, {}, "", 0, 0, 0, false, AllocStatus::OK, AllocStatus::OK, {{2, 2}, {5, 3}}, {2, 3}},
    };
    SwapInTestHelper(sequencialAllocateTestData, sequencialSwapInTestData);
}

std::vector<ConsistencyTestData> generateToken(int dataSize, int prefillSize, int decodeSize, size_t blockSize,
                                               size_t cpuNum, size_t npuNum, int repeat, int stop)
{
    std::vector<ConsistencyTestData> ret;
    std::vector<TokenId> prefillTokens;
    std::vector<TokenId> decodeTokens;
    std::vector<TokenId> moreDecodeTokens;

    while (repeat < stop) {
        for (int i = 0; i < dataSize; i++) {
            for (int j = 0; j < prefillSize; j++) {
                prefillTokens.push_back(i * dataSize + j);
            }
        }
        // decode
        for (int i = 0; i < dataSize; i++) {
            for (int j = 0; j < decodeSize; j++) {
                decodeTokens.push_back(i * dataSize + j + repeat * 1000); // some random token
            }
        }
        for (int i = 0; i < dataSize; i++) {
            for (int j = 0; j < decodeSize; j++) {
                moreDecodeTokens.push_back(i * dataSize + j * 10000 + repeat * 1000); // some random token
                ret.push_back(ConsistencyTestData{
                    i * 10000000 + j * 10000 + repeat * 100, prefillTokens, decodeTokens, moreDecodeTokens,
                    "Req:" + std::to_string(i * 10000000 + j * 10000 + repeat * 100), blockSize, cpuNum, npuNum, true});
            }
        }
        repeat++;
    }
    std::unordered_map<SequenceId, int> sequenceOccurence;
    for (auto data : ret) {
        sequenceOccurence[data.seqId]++;
        assert(sequenceOccurence[data.seqId] == 1);
    }
    return ret;
}

class CacheEngine {
public:
    explicit CacheEngine(std::shared_ptr<SelfAttnBlockManager> blockManagerPtr) : blockManagerPtr(blockManagerPtr)
    {
        // 获取一些私有变量
        auto cpuNpuAllocator = static_cast<CpuNpuBlockAllocator *>(blockManagerPtr.get()->blockAllocator_.get());
        auto npuPrefixAllocator = cpuNpuAllocator->GetAllocator(DeviceType::NPU);
        auto cpuPrefixAllocator = cpuNpuAllocator->GetAllocator(DeviceType::CPU);
        auto blockPool = static_cast<PrefixCacheBlockAllocator *>(npuPrefixAllocator.get())->blockObjPool_.get();

        RefCounterProtocol *npuRefCounterP =
            static_cast<PrefixCacheBlockAllocator *>(npuPrefixAllocator.get())->refCounter_.get();
        RefCounterProtocol *cpuRefCounterP =
            static_cast<PrefixCacheBlockAllocator *>(cpuPrefixAllocator.get())->refCounter_.get();
        npuRefCountsPtr = &(static_cast<RefCounter *>(npuRefCounterP)->refCounts_);
        cpuRefCountsPtr = &(static_cast<RefCounter *>(cpuRefCounterP)->refCounts_);
        numNpu = blockManagerPtr->npuBlockNum_;
        numCpu = blockManagerPtr->cpuBlockNum_;
        npuCache.resize(numNpu);
        cpuCache.resize(numCpu);
    };
    ~CacheEngine() = default;
    // 在blockmanager调用swapout，swapin后，本函数被调用，更新npucache，cpucache
    void swapBlocks(std::vector<std::pair<PhysicalBlockId, PhysicalBlockId>> &physicalBlockIdMapping, SequenceSPtr seq,
                    bool fromNpuToCpu);

    // 在blockmanager调用append后，本函数被调用，将seq中的tokenid更新到npucache
    void addTokensToBlocks(SequenceSPtr seq);

    // 在blockmanager调用free后，本函数被调用，释放npucache
    void freeBlocks(std::vector<BlockId> &blockIds);

    // 在执行完对一组数据的操作后，检查所有sequence的每个block中的tokenId应该和cacheEngine中的tokenId一致
    void checkNpuCacheConsistency(std::vector<SequenceGroupSPtr> seqGroupPtrs);

    //存储npu中blockId和tokenId的映射
    std::vector<std::vector<TokenId>> npuCache;

    //存储cpu中blockId和tokenId的映射
    std::vector<std::vector<TokenId>> cpuCache;
    std::shared_ptr<SelfAttnBlockManager> blockManagerPtr;
    std::unordered_map<BlockId, RefCount> *npuRefCountsPtr;
    std::unordered_map<BlockId, RefCount> *cpuRefCountsPtr;
    int numNpu;
    int numCpu;
};

void CacheEngine::swapBlocks(std::vector<std::pair<PhysicalBlockId, PhysicalBlockId>> &physicalBlockIdMapping,
                             SequenceSPtr seq, bool fromNpuToCpu)
{
    std::vector<BlockObjSPtr> blocks = blockManagerPtr->seqId2BlockTable_.at(seq->seqId_).GetBlockObjs();
    for (int i = 0; i < blocks.size(); i++) {
        auto mapping = physicalBlockIdMapping[i];
        auto blockObj = blocks[i];
        if (fromNpuToCpu) { // swapout
            BlockId cpuBlockId = mapping.second;
            cpuCache[cpuBlockId].clear();
            cpuCache[cpuBlockId] = static_cast<PrefixCachingBlockObj *>(blockObj.get())->tokenIds_;
        } else { // swapin
            BlockId npuBlockId = mapping.second;
            npuCache[npuBlockId].clear();
            npuCache[npuBlockId] = static_cast<PrefixCachingBlockObj *>(blockObj.get())->tokenIds_;
        }
    }
};

void CacheEngine::addTokensToBlocks(SequenceSPtr seq)
{
    std::vector<BlockObjSPtr> blocks = blockManagerPtr->seqId2BlockTable_.at(seq->seqId_).GetBlockObjs();
    for (auto block : blocks) {
        auto blockObj = static_cast<PrefixCachingBlockObj *>(block.get());
        std::vector<TokenId> tokenIds = blockObj->tokenIds_;
        BlockId blockId = blockObj->blockId_;
        npuCache[blockId].clear();
        npuCache[blockId] = tokenIds;
    }
};

void CacheEngine::checkNpuCacheConsistency(std::vector<SequenceGroupSPtr> seqGroupPtrs)
{
    for (auto groupPtr : seqGroupPtrs) {
        auto seq = groupPtr->firstSeq;
        auto blockTable = blockManagerPtr->seqId2BlockTable_[seq->seqId_];
        auto blocks = blockTable.GetBlockObjs();
        for (auto block : blocks) {
            auto blockObj = static_cast<PrefixCachingBlockObj *>(block.get());
            BlockId blockId = blockObj->blockId_;
            EXPECT_EQ(npuCache[blockId], blockObj->tokenIds_);
        }
    }
}

void ConsistencyTestHelper(const std::vector<ConsistencyTestData> &sequencialTestData)
{
    size_t blockSize = sequencialTestData[0].blockSize;
    size_t cpuBlockNum = sequencialTestData[0].cpuBlockNum;
    size_t npuBlockNum = sequencialTestData[0].npuBlockNum;
    bool enableCaching = sequencialTestData[0].enableCaching;
    size_t reservedBlockNum = 0;
    size_t speculativeSlots = 0;
    std::shared_ptr<SamplingParams> sampling = nullptr;
    BlockManagerConfig config{blockSize, cpuBlockNum, npuBlockNum, reservedBlockNum, speculativeSlots, enableCaching};
    SelfAttnBlockManager blockManager(config);
    std::shared_ptr<SelfAttnBlockManager> blockManagerPtr = std::make_shared<SelfAttnBlockManager>(blockManager);
    CacheEngine cacheEngine = CacheEngine(blockManagerPtr);
    std::vector<SequenceGroupSPtr> seqGroupPtrs;
    std::vector<SequenceSPtr> seqPtrs;
    int extractFactor = 4;

    int numTestData = sequencialTestData.size();
    int groupSize = 64;
    int currPos = 0;
    // 将数据分成多个组进行测试，每组最多64个数据
    while (currPos < numTestData) {
        int newlyAddNumData = 0;
        for (int index = currPos; index < currPos + groupSize && index < numTestData; index++) {
            ConsistencyTestData allocateTestData = sequencialTestData[index];
            SequenceSPtr seqPtr = std::make_shared<Sequence>(allocateTestData.seqId, allocateTestData.blockSize,
                                                             allocateTestData.inputs);
            std::vector<std::shared_ptr<Sequence>> seqs = {seqPtr};
            RequestId thisRequestId = allocateTestData.requestId;
            SequenceGroupSPtr groupPtr = std::make_shared<SequenceGroup>(thisRequestId, seqs, sampling);

            AllocStatus canAllocate = blockManagerPtr->CanAllocate(groupPtr);
            if (canAllocate != AllocStatus::OK) {
                throw std::runtime_error("cant allocate");
            }
            bool res = blockManagerPtr->Allocate(groupPtr); // CHECK: Allocate
            cacheEngine.addTokensToBlocks(seqPtr);
            seqGroupPtrs.push_back(groupPtr);
            seqPtrs.push_back(seqPtr);
            newlyAddNumData++;
        }
        cacheEngine.checkNpuCacheConsistency(seqGroupPtrs);

        // swapout
        for (int index = currPos; index < currPos + newlyAddNumData; index++) {
            auto seqPtr = seqPtrs[index];
            auto groupPtr = seqGroupPtrs[index];
            // 设置序列状态为 RUNNING
            seqPtr->status_ = SequenceStatus::RUNNING;
            bool canSwapOut = blockManagerPtr->CanSwapOut(groupPtr);
            if (!canSwapOut) {
                throw std::runtime_error("cant swapout");
            }
            auto physicalBlockIdMapping = blockManagerPtr->SwapOut(groupPtr);
            cacheEngine.swapBlocks(physicalBlockIdMapping, seqPtr, true);
        }

        for (int index = currPos; index < currPos + newlyAddNumData; index++) {
            auto seqPtr = seqPtrs[index];
            auto groupPtr = seqGroupPtrs[index];
            // 设置序列状态为 SWAPPED
            seqPtr->status_ = SequenceStatus::SWAPPED;

            AllocStatus canSwapIn = blockManagerPtr->CanSwapIn(groupPtr, 0);
            if (canSwapIn != AllocStatus::OK) {
                throw std::runtime_error("cant swapin");
            }
            auto physicalBlockIdMapping = blockManagerPtr->SwapIn(groupPtr);
            cacheEngine.swapBlocks(physicalBlockIdMapping, seqPtr, false);
            auto blockids = blockManagerPtr->GetBlockIds(seqPtr->seqId_);
        }
        cacheEngine.checkNpuCacheConsistency(seqGroupPtrs);

        for (int index = currPos; index < currPos + newlyAddNumData; index++) {
            auto seqPtr = seqPtrs[index];
            auto groupPtr = seqGroupPtrs[index];
            // 设置序列状态为 RUNNING
            seqPtr->status_ = SequenceStatus::RUNNING;
            seqPtr->data_.outputTokenIds.insert(seqPtr->data_.outputTokenIds.end(),
                                                sequencialTestData[index].tokensToAppend.begin(),
                                                sequencialTestData[index].tokensToAppend.end());
            bool canAppend = blockManagerPtr->CanAppendSlot(groupPtr);
            if (!canAppend) {
                throw std::runtime_error("cant append");
            }
            blockManagerPtr->AppendSlot(seqPtr);
            cacheEngine.addTokensToBlocks(seqPtr);
        }
        cacheEngine.checkNpuCacheConsistency(seqGroupPtrs);

        for (int index = currPos; index < currPos + newlyAddNumData; index++) {
            auto seqPtr = seqPtrs[index];
            auto groupPtr = seqGroupPtrs[index];
            // 设置序列状态为 RUNNING
            seqPtr->status_ = SequenceStatus::RUNNING;
            bool canSwapOut = blockManagerPtr->CanSwapOut(groupPtr);
            if (!canSwapOut) {
                throw std::runtime_error("cant swapout");
            }
            auto physicalBlockIdMapping = blockManagerPtr->SwapOut(groupPtr);
            cacheEngine.swapBlocks(physicalBlockIdMapping, seqPtr, true);
        }

        for (int index = currPos; index < currPos + newlyAddNumData; index++) {
            auto seqPtr = seqPtrs[index];
            auto groupPtr = seqGroupPtrs[index];
            // 设置序列状态为 SWAPPED
            seqPtr->status_ = SequenceStatus::SWAPPED;

            AllocStatus canSwapIn = blockManagerPtr->CanSwapIn(groupPtr, 0);
            if (canSwapIn != AllocStatus::OK) {
                throw std::runtime_error("cant swapin");
            }
            auto physicalBlockIdMapping = blockManagerPtr->SwapIn(groupPtr);
            cacheEngine.swapBlocks(physicalBlockIdMapping, seqPtr, false);
        }
        cacheEngine.checkNpuCacheConsistency(seqGroupPtrs);

        for (int index = currPos; index < currPos + newlyAddNumData; index++) {
            auto seqPtr = seqPtrs[index];
            auto groupPtr = seqGroupPtrs[index];
            // 设置序列状态为 RUNNING
            seqPtr->status_ = SequenceStatus::RUNNING;
            seqPtr->data_.outputTokenIds.insert(seqPtr->data_.outputTokenIds.end(),
                                                sequencialTestData[index].moreTokensToAppend.begin(),
                                                sequencialTestData[index].moreTokensToAppend.end());
            bool canAppend = blockManagerPtr->CanAppendSlot(groupPtr);
            if (!canAppend) {
                throw std::runtime_error("cant append");
            }
            blockManagerPtr->AppendSlot(seqPtr);
            cacheEngine.addTokensToBlocks(seqPtr);
        }
        cacheEngine.checkNpuCacheConsistency(seqGroupPtrs);

        // partial free
        for (int index = currPos; index < currPos + newlyAddNumData / 2; index++) {
            auto seqPtr = seqPtrs[index];
            auto groupPtr = seqGroupPtrs[index];
            seqPtr->status_ = SequenceStatus::RUNNING;
            std::vector<BlockId> blockIds = blockManagerPtr->seqId2BlockTable_.at(seqPtr->seqId_).GetBlockIds();
            std::vector<BlockObjSPtr> blockObjs = blockManagerPtr->seqId2BlockTable_.at(seqPtr->seqId_).GetBlockObjs();
            blockManagerPtr->GetCommonComputedBlockIds(groupPtr->GetSequences(SequenceStatus::RUNNING));
            blockManagerPtr->Free(seqPtr->seqId_);
        }
        cacheEngine.checkNpuCacheConsistency(seqGroupPtrs);

        currPos += newlyAddNumData;
    }
}
/**
 * 进行多轮append，swapout，swapin，free，每轮每次操作后检测是否踩踏
 */
// 资源充足情况下测试
TEST(PrefixConsistencyTest, lessRandomData)
{
    const std::vector<ConsistencyTestData> sequencialTestData = generateToken(10, 25, 25, 16, 1000, 1000, 0, 2);
    ConsistencyTestHelper(sequencialTestData);
}

TEST(PrefixConsistencyTest, moreRandomData)
{
    const std::vector<ConsistencyTestData> sequencialTestData = generateToken(10, 25, 25, 16, 500, 500, 0, 2);
    ConsistencyTestHelper(sequencialTestData);
}

} // namespace mindie_llm