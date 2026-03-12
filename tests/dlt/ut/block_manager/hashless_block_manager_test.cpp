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
#include "self_attn_block_manager.h"
#include "sequence_group.h"
#include "hashless_block_allocator.h"

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

/**
 * Swapout测试场景1：进行一次allocate+swapout
 * 预期结果：
 * 1. allocate+swapout成功，最后blocks都在CPU上，对应的block数量、blockID正确、swapout返回的mapping正确
 * 2. cpu空间不够，swapout失败，swapout阻塞
 */
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

const SwapOutTestData swapOutTestData[] = {
    // 单block场景
    {0, {1, 2, 3, 4, 5, 6, 7, 8}, "Req:0", 8, 10, 10, false, AllocStatus::OK, true, {{0, 0}}, {10}},
    // 多block场景
    {0,
     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
     "Req:0",
     8,
     10,
     10,
     false,
     AllocStatus::OK,
     true,
     {{0, 0}, {1, 1}, {2, 2}},
     {10, 11, 12}},
    // cpu空间不够，不能swapout，阻塞
    {0, {1, 2, 3, 4, 5, 6, 7, 8, 9}, "Req:0", 8, 0, 9, false, AllocStatus::OK, false, {}, {}},
};

class SwapOutTestClass : public ::testing::TestWithParam<SwapOutTestData> {
protected:
    std::shared_ptr<SamplingParams> sampling;
    size_t reservedBlockNum = 0;
    size_t speculativeSlots = 0;
};

TEST_P(SwapOutTestClass, ExpectOneAllocateAndSwapOutSuccessHavingEnoughCpuAndFailNotHavingEnoughCpu)
{
    SwapOutTestData test = GetParam();
    BlockManagerConfig config{test.blockSize,   test.cpuBlockNum, test.npuBlockNum,
                              reservedBlockNum, speculativeSlots, test.enableCaching};
    SelfAttnBlockManager blockManager(config);
    //手动求出需要几个blocks
    int numBlocks = std::ceil(test.inputs.size() * 1.0 / test.blockSize);
    //初始化seq group
    SequenceSPtr seqPtr = std::make_shared<Sequence>(test.seqId, test.blockSize, test.inputs);
    std::vector<std::shared_ptr<Sequence>> seqs = {seqPtr};
    SequenceGroupSPtr groupPtr = std::make_shared<SequenceGroup>(test.requestId, seqs, sampling);

    AllocStatus canAllocate = blockManager.CanAllocate(groupPtr);
    EXPECT_EQ(canAllocate, test.canAllocate);

    EXPECT_EQ(blockManager.GetNumFreeCpuBlocks(), test.cpuBlockNum);
    EXPECT_EQ(blockManager.GetNumFreeNpuBlocks(), test.npuBlockNum);

    blockManager.Allocate(groupPtr);

    EXPECT_EQ(blockManager.GetNumFreeCpuBlocks(), test.cpuBlockNum);
    EXPECT_EQ(blockManager.GetNumFreeNpuBlocks(), test.npuBlockNum - numBlocks);

    seqPtr->status_ = SequenceStatus::RUNNING;

    bool canSwapOut = blockManager.CanSwapOut(groupPtr);
    EXPECT_EQ(canSwapOut, test.canSwapOut);
    //如果阻塞了，就不进行后面判断
    if (canSwapOut == false) {
        return;
    }
    auto physicalBlockIdMapping = blockManager.SwapOut(groupPtr);
    EXPECT_EQ(physicalBlockIdMapping, test.physicalBlockIdMappingExpected);

    EXPECT_EQ(blockManager.GetNumFreeCpuBlocks(), test.cpuBlockNum - numBlocks);
    EXPECT_EQ(blockManager.GetNumFreeNpuBlocks(), test.npuBlockNum);

    auto blockIds = blockManager.GetBlockIds(seqPtr->seqId_);
    ASSERT_EQ(blockIds.size(), 1u);
    EXPECT_EQ(blockIds[0], test.blockIdsExpected);
};

/**
 * Swapout测试场景2：allocate满整个NPU blocks+全部swapout
 * 预期结果：
 * 1. allocate+swapout成功，最后blocks都在CPU上，对应的block数量、blockID正确、swapout返回的mapping正确
 */
const std::vector<SwapOutTestData> sequencialSwapOutTestData = {
    //在这个测试场景，只有SwapOutTestData最后三个字段需要比较
    SwapOutTestData{-1, {}, "", 0, 0, 0, false, AllocStatus::OK, true, {{0, 0}}, {3}},
    SwapOutTestData{-1, {}, "", 0, 0, 0, false, AllocStatus::OK, true, {{1, 1}}, {4}},
    // swapout前后block都指向同一个物理块2，5表明块在CPU上
    SwapOutTestData{-1, {}, "", 0, 0, 0, false, AllocStatus::OK, true, {{2, 2}}, {5}},
};
TEST_P(SwapOutTestClass, ExpectSuccessHavingMultipleAllocateAndSwapOut)
{
    const std::vector<AllocateTestData> sequencialAllocateTestData = {
        AllocateTestData{10, {1, 2, 3, 4, 5, 6, 7, 8}, "Req:0", 8, 3, 3, false, AllocStatus::OK},
        AllocateTestData{21, {11, 12, 13, 14, 15, 16, 17, 18}, "Req:1", 8, 3, 3, false, AllocStatus::OK},
        AllocateTestData{36, {21, 22, 23, 24, 25, 26}, "Req:2", 8, 3, 3, false, AllocStatus::OK},
    };
    size_t blockSize = sequencialAllocateTestData[0].blockSize;
    size_t cpuBlockNum = sequencialAllocateTestData[0].cpuBlockNum;
    size_t npuBlockNum = sequencialAllocateTestData[0].npuBlockNum;
    bool enableCaching = sequencialAllocateTestData[0].enableCaching;
    // 初始化测试数据
    BlockManagerConfig config{blockSize, cpuBlockNum, npuBlockNum, reservedBlockNum, speculativeSlots, enableCaching};
    SelfAttnBlockManager blockManager(config);

    std::vector<SequenceGroupSPtr> seqGroupPtrs;
    std::vector<SequenceSPtr> seqPtrs;
    std::vector<int> AllocatedNumBlocks;
    int numAllAllocatedBlocks = 0;
    int currNumFreeCpuBlocks = cpuBlockNum;
    int currNumFreeNpuBlocks = npuBlockNum;
    for (auto &allocateTestData : sequencialAllocateTestData) {

        SequenceSPtr seqPtr = std::make_shared<Sequence>(allocateTestData.seqId, allocateTestData.blockSize,
                                                         allocateTestData.inputs);
        std::vector<std::shared_ptr<Sequence>> seqs = {seqPtr};
        RequestId thisRequestId = allocateTestData.requestId;
        SequenceGroupSPtr groupPtr = std::make_shared<SequenceGroup>(thisRequestId, seqs, sampling);
        seqGroupPtrs.push_back(groupPtr);
        seqPtrs.push_back(seqPtr);

        EXPECT_EQ(blockManager.GetNumFreeCpuBlocks(), currNumFreeCpuBlocks);
        EXPECT_EQ(blockManager.GetNumFreeNpuBlocks(), currNumFreeNpuBlocks);

        AllocStatus canAllocate = blockManager.CanAllocate(groupPtr);
        EXPECT_EQ(canAllocate, allocateTestData.canAllocate); // CHECK: CanAllocate
        EXPECT_EQ(blockManager.GetNumFreeCpuBlocks(), currNumFreeCpuBlocks);
        EXPECT_EQ(blockManager.GetNumFreeNpuBlocks(), currNumFreeNpuBlocks);

        blockManager.Allocate(groupPtr); // CHECK: Allocate
        //手动求出需要几个blocks
        int numAllocatedBlocks = std::ceil(allocateTestData.inputs.size() * 1.0 / allocateTestData.blockSize);
        AllocatedNumBlocks.push_back(numAllocatedBlocks);
        numAllAllocatedBlocks += numAllocatedBlocks;
        currNumFreeNpuBlocks -= numAllocatedBlocks;

        EXPECT_EQ(blockManager.GetNumFreeCpuBlocks(), currNumFreeCpuBlocks);
        EXPECT_EQ(blockManager.GetNumFreeNpuBlocks(), currNumFreeNpuBlocks);
    }
    int i = 0;
    for (const auto &test : sequencialSwapOutTestData) {

        auto seqPtr = seqPtrs[i];
        auto groupPtr = seqGroupPtrs[i];
        // 设置序列状态为 RUNNING
        seqPtr->status_ = SequenceStatus::RUNNING;

        EXPECT_EQ(blockManager.GetNumFreeCpuBlocks(), currNumFreeCpuBlocks);
        EXPECT_EQ(blockManager.GetNumFreeNpuBlocks(), currNumFreeNpuBlocks);

        bool canSwapOut = blockManager.CanSwapOut(groupPtr);
        EXPECT_EQ(canSwapOut, test.canSwapOut); // 检查 CanSwapOut 是否符合预期

        EXPECT_EQ(blockManager.GetNumFreeCpuBlocks(), currNumFreeCpuBlocks);
        EXPECT_EQ(blockManager.GetNumFreeNpuBlocks(), currNumFreeNpuBlocks);

        auto physicalBlockIdMapping = blockManager.SwapOut(groupPtr);
        //手动求出blocks数量变化
        currNumFreeCpuBlocks -= AllocatedNumBlocks[i];
        currNumFreeNpuBlocks += AllocatedNumBlocks[i];
        EXPECT_EQ(physicalBlockIdMapping, test.physicalBlockIdMappingExpected); // 检查物理块映射是否符合预期

        EXPECT_EQ(blockManager.GetNumFreeCpuBlocks(), currNumFreeCpuBlocks);
        EXPECT_EQ(blockManager.GetNumFreeNpuBlocks(), currNumFreeNpuBlocks);

        auto blockIds = blockManager.GetBlockIds(seqPtr->seqId_);
        ASSERT_EQ(blockIds.size(), 1u);
        EXPECT_EQ(blockIds[0], test.blockIdsExpected); // 检查块 ID 是否符合预期
        numAllAllocatedBlocks -= AllocatedNumBlocks[i];
        i++;
    }
}
INSTANTIATE_TEST_SUITE_P(SwapOutTestSuite, SwapOutTestClass, ::testing::ValuesIn(swapOutTestData));

/**
 * Swapin测试场景1：进行一次allocate+swapout+swapin+free
 * 预期结果：
 * 1. allocate+swapout+swapin+free成功，每步对应的block数量、blockID正确、swapin返回的mapping正确
 */
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
    std::vector<BlockId> blockIdsExpectedAfterSwapIn;
};

SwapInTestData swapInTestData[] = {
    //单block场景
    {0, {1, 2, 3, 4, 5, 6, 7, 8}, "Req:0", 8, 1, 1, false, AllocStatus::OK, AllocStatus::OK, {{0, 0}}, {0}},
    //多block场景
    {0,
     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 23},
     "Req:0",
     8,
     2,
     2,
     false,
     AllocStatus::OK,
     AllocStatus::OK,
     {{0, 0}, {1, 1}},
     {0, 1}}};

class SwapInTestClass : public ::testing::TestWithParam<SwapInTestData> {
protected:
    std::vector<std::shared_ptr<Sequence>> seqs;
    std::shared_ptr<SamplingParams> sampling;
    size_t reservedBlockNum = 0;
    size_t speculativeSlots = 0;
};

TEST_P(SwapInTestClass, ExpectSuccessHavingOneAllocateAndSwapOutAndSwapInAndFree)
{
    SwapInTestData test = GetParam();
    BlockManagerConfig config{test.blockSize,   test.cpuBlockNum, test.npuBlockNum,
                              reservedBlockNum, speculativeSlots, test.enableCaching};
    SelfAttnBlockManager blockManager(config);
    int numBlocks = std::ceil(test.inputs.size() * 1.0 / test.blockSize);
    SequenceSPtr seqPtr = std::make_shared<Sequence>(test.seqId, test.blockSize, test.inputs);
    std::vector<std::shared_ptr<Sequence>> seqs = {seqPtr};
    SequenceGroupSPtr groupPtr = std::make_shared<SequenceGroup>(test.requestId, seqs, sampling);

    AllocStatus canAllocate = blockManager.CanAllocate(groupPtr);
    blockManager.Allocate(groupPtr);
    seqPtr->status_ = SequenceStatus::RUNNING;
    bool canSwapOut = blockManager.CanSwapOut(groupPtr);
    auto physicalBlockIdMapping = blockManager.SwapOut(groupPtr);

    seqPtr->status_ = SequenceStatus::SWAPPED;

    EXPECT_EQ(blockManager.GetNumFreeCpuBlocks(), test.cpuBlockNum - numBlocks);
    EXPECT_EQ(blockManager.GetNumFreeNpuBlocks(), test.npuBlockNum);

    AllocStatus canSwapIn = blockManager.CanSwapIn(groupPtr, config.speculativeSlots);
    EXPECT_EQ(canSwapIn, test.canSwapIn);
    EXPECT_EQ(blockManager.GetNumFreeCpuBlocks(), test.cpuBlockNum - numBlocks);
    EXPECT_EQ(blockManager.GetNumFreeNpuBlocks(), test.npuBlockNum);

    physicalBlockIdMapping = blockManager.SwapIn(groupPtr);
    EXPECT_EQ(physicalBlockIdMapping, test.physicalBlockIdMappingExpected);
    EXPECT_EQ(blockManager.GetNumFreeCpuBlocks(), test.cpuBlockNum);
    EXPECT_EQ(blockManager.GetNumFreeNpuBlocks(), test.npuBlockNum - numBlocks);

    auto blockIds = blockManager.GetBlockIds(seqPtr->seqId_);
    ASSERT_EQ(blockIds.size(), 1u);
    EXPECT_EQ(blockIds[0], test.blockIdsExpectedAfterSwapIn);

    blockManager.Free(seqPtr->seqId_);
    EXPECT_EQ(blockManager.GetNumFreeCpuBlocks(), test.cpuBlockNum);
    EXPECT_EQ(blockManager.GetNumFreeNpuBlocks(), test.npuBlockNum);
};
INSTANTIATE_TEST_SUITE_P(SwapInTestSuite, SwapInTestClass, ::testing::ValuesIn(swapInTestData));

/**
 * Swapin测试场景2：allocate满整个NPU+全部swapout+allocate一些空间在NPU占用+swapin小块+swapin大块+free
 * 预期结果：
 * 1. swapin小块成功，因为NPU还有空间每步对应的block数量、blockID正确、swapin返回的mapping正确
 * 2. swapout大块失败，swapin阻塞
 */
class SwapTestClass : public ::testing::TestWithParam<SwapInTestData> {
protected:
    std::shared_ptr<SamplingParams> sampling = nullptr;
    size_t reservedBlockNum = 0;
    size_t speculativeSlots = 0;
};
const std::vector<SwapInTestData> sequencialSwapInTestData = {
    //在这个测试场景，只有SwapInTestData最后三个字段需要比较
    //这个seq最先被换入，NPU空间足够，swapin成功
    SwapInTestData{-1, {}, "", 0, 0, 0, false, AllocStatus::OK, AllocStatus::OK, {{0, 2}}, {2}},
    //这个seq最后被换入，NPU空间不够，swapin失败
    SwapInTestData{-1, {}, "", 0, 0, 0, false, AllocStatus::OK, AllocStatus::LATER, {{}}, {}},
};
TEST_P(SwapTestClass, ExpectSwapInSuccessHavingEnoughNpuAndSwapInFailNotHavingEnoughNpu)
{
    const std::vector<AllocateTestData> sequencialAllocateTestData = {
        AllocateTestData{0, {1, 2, 3, 4, 5, 6, 7, 7}, "Req:0", 8, 3, 3, false, AllocStatus::OK},
        AllocateTestData{1, {11, 22, 33, 44, 55, 66, 77, 88, 21, 22, 23}, "Req:1", 8, 3, 3, false, AllocStatus::OK},
    };
    size_t blockSize = sequencialAllocateTestData[0].blockSize;
    size_t cpuBlockNum = sequencialAllocateTestData[0].cpuBlockNum;
    size_t npuBlockNum = sequencialAllocateTestData[0].npuBlockNum;
    bool enableCaching = sequencialAllocateTestData[0].enableCaching;

    BlockManagerConfig config{blockSize, cpuBlockNum, npuBlockNum, reservedBlockNum, speculativeSlots, enableCaching};
    SelfAttnBlockManager blockManager(config);

    std::vector<SequenceGroupSPtr> seqGroupPtrs;
    std::vector<SequenceSPtr> seqPtrs;
    std::vector<int> AllocatedNumBlocks;
    int numAllAllocatedBlocks = 0;
    int currNumFreeCpuBlocks = cpuBlockNum;
    int currNumFreeNpuBlocks = npuBlockNum;
    // allocate先占满NPU
    for (auto &allocateTestData : sequencialAllocateTestData) {
        SequenceSPtr seqPtr = std::make_shared<Sequence>(allocateTestData.seqId, allocateTestData.blockSize,
                                                         allocateTestData.inputs);
        std::vector<std::shared_ptr<Sequence>> seqs = {seqPtr};
        RequestId thisRequestId = allocateTestData.requestId;
        SequenceGroupSPtr groupPtr = std::make_shared<SequenceGroup>(thisRequestId, seqs, sampling);
        seqGroupPtrs.push_back(groupPtr);
        seqPtrs.push_back(seqPtr);

        AllocStatus canAllocate = blockManager.CanAllocate(groupPtr);
        blockManager.Allocate(groupPtr);

        int numAllocatedBlocks = std::ceil(allocateTestData.inputs.size() * 1.0 / allocateTestData.blockSize);
        AllocatedNumBlocks.push_back(numAllocatedBlocks);
        numAllAllocatedBlocks += numAllocatedBlocks;
        currNumFreeNpuBlocks -= numAllocatedBlocks;
    }
    // 全部swapout
    for (int i = 0; i < sequencialAllocateTestData.size(); i++) {
        auto seqPtr = seqPtrs[i];
        auto groupPtr = seqGroupPtrs[i];
        // 设置序列状态为 RUNNING
        seqPtr->status_ = SequenceStatus::RUNNING;
        bool canSwapOut = blockManager.CanSwapOut(groupPtr);

        auto physicalBlockIdMapping = blockManager.SwapOut(groupPtr);
        currNumFreeCpuBlocks -= AllocatedNumBlocks[i];
        currNumFreeNpuBlocks += AllocatedNumBlocks[i];
    }
    // allocate一个序列，之后NPU还有两个块空闲
    AllocateTestData newAllocateTestData =
        AllocateTestData{2, {1, 2, 3, 4, 5, 11, 22, 33, 44, 55, 66, 77, 88}, "Req:2", 8, 3, 3, false, AllocStatus::OK};
    SequenceSPtr seqPtr = std::make_shared<Sequence>(newAllocateTestData.seqId, newAllocateTestData.blockSize,
                                                     newAllocateTestData.inputs);
    std::vector<std::shared_ptr<Sequence>> seqs = {seqPtr};
    RequestId thisRequestId = newAllocateTestData.requestId;
    SequenceGroupSPtr groupPtr = std::make_shared<SequenceGroup>(thisRequestId, seqs, sampling);

    AllocStatus canAllocate = blockManager.CanAllocate(groupPtr);
    blockManager.Allocate(groupPtr);
    currNumFreeNpuBlocks -= std::ceil(newAllocateTestData.inputs.size() * 1.0 / newAllocateTestData.blockSize);
    // 再swapin
    for (int i = 0; i < sequencialSwapInTestData.size(); i++) {
        auto seqPtr = seqPtrs[i];
        auto groupPtr = seqGroupPtrs[i];
        // 设置序列状态为 SWAPPED
        seqPtr->status_ = SequenceStatus::SWAPPED;
        EXPECT_EQ(blockManager.GetNumFreeCpuBlocks(), currNumFreeCpuBlocks);
        EXPECT_EQ(blockManager.GetNumFreeNpuBlocks(), currNumFreeNpuBlocks);
        AllocStatus canSwapIn = blockManager.CanSwapIn(groupPtr, speculativeSlots);
        EXPECT_EQ(canSwapIn, sequencialSwapInTestData[i].canSwapIn);

        std::vector<std::pair<PhysicalBlockId, PhysicalBlockId>> physicalBlockIdMapping;
        if (canSwapIn == AllocStatus::LATER) {
            // NPU没有空间，BlockAllocator Error: No Free Blocks! 不用进行之后的判断
            EXPECT_THROW({ physicalBlockIdMapping = blockManager.SwapIn(groupPtr); }, std::runtime_error);
            continue;
        } else {
            // NPU有空间，swapin成功
            physicalBlockIdMapping = blockManager.SwapIn(groupPtr);
        }
        currNumFreeCpuBlocks += AllocatedNumBlocks[i];
        currNumFreeNpuBlocks -= AllocatedNumBlocks[i];

        EXPECT_EQ(physicalBlockIdMapping, sequencialSwapInTestData[i].physicalBlockIdMappingExpected);
        EXPECT_EQ(blockManager.GetNumFreeCpuBlocks(), currNumFreeCpuBlocks);
        EXPECT_EQ(blockManager.GetNumFreeNpuBlocks(), currNumFreeNpuBlocks);

        auto blockIds = blockManager.GetBlockIds(seqPtr->seqId_);
        ASSERT_EQ(blockIds.size(), 1u);
        EXPECT_EQ(blockIds[0], sequencialSwapInTestData[i].blockIdsExpectedAfterSwapIn);
    }
}
INSTANTIATE_TEST_SUITE_P(SwapTestSuite, SwapTestClass, ::testing::ValuesIn(sequencialSwapInTestData));

/**
 * Swapin测试场景3：allocate满整个NPU+全部swapout+按swapout顺序反向swapin（为了确保swap后blockID mapping正确）+free
 * 预期结果：
 * 1. 成功，每步对应的block数量、blockID正确、swapin返回的mapping正确
 */
TEST(SwapTestClass, ExpectSuccessWhenSwapOutAndReverselySwapIn)
{
    const std::vector<SwapInTestData> sequencialSwapInTestData = {
        //这个seq最后被换入，从CPU的第0个块换到NPU的第2个块，最后物理blockID是2
        SwapInTestData{-1, {}, "Req:0", 0, 0, 0, false, AllocStatus::OK, AllocStatus::OK, {{0, 2}}, {2}},
        SwapInTestData{-1, {}, "Req:1", 0, 0, 0, false, AllocStatus::OK, AllocStatus::OK, {{1, 1}}, {1}},
        //这个seq最先被换入，从CPU的第2个块换到NPU的第0个块，最后物理blockID是0
        SwapInTestData{-1, {}, "Req:2", 0, 0, 0, false, AllocStatus::OK, AllocStatus::OK, {{2, 0}}, {0}},
    };

    const std::vector<AllocateTestData> sequencialAllocateTestData = {
        AllocateTestData{0, {1, 2, 3, 4, 5, 6, 7, 8}, "Req:0", 8, 3, 3, false, AllocStatus::OK},
        AllocateTestData{1, {11, 12, 13, 14, 15, 16, 17}, "Req:1", 8, 3, 3, false, AllocStatus::OK},
        AllocateTestData{2, {21, 22, 23, 24, 25, 26, 27, 28}, "Req:2", 8, 3, 3, false, AllocStatus::OK},
    };
    size_t blockSize = sequencialAllocateTestData[0].blockSize;
    size_t cpuBlockNum = sequencialAllocateTestData[0].cpuBlockNum;
    size_t npuBlockNum = sequencialAllocateTestData[0].npuBlockNum;
    bool enableCaching = sequencialAllocateTestData[0].enableCaching;
    size_t reservedBlockNum = 0;
    size_t speculativeSlots = 0;
    std::shared_ptr<SamplingParams> sampling = nullptr;
    BlockManagerConfig config{blockSize, cpuBlockNum, npuBlockNum, reservedBlockNum, speculativeSlots, enableCaching};
    SelfAttnBlockManager blockManager(config);

    std::vector<SequenceGroupSPtr> seqGroupPtrs;
    std::vector<SequenceSPtr> seqPtrs;
    std::vector<int> AllocatedNumBlocks;
    int numAllAllocatedBlocks = 0;
    int currNumFreeCpuBlocks = cpuBlockNum;
    int currNumFreeNpuBlocks = npuBlockNum;
    // allocate所有
    for (auto &allocateTestData : sequencialAllocateTestData) {
        SequenceSPtr seqPtr = std::make_shared<Sequence>(allocateTestData.seqId, allocateTestData.blockSize,
                                                         allocateTestData.inputs);
        std::vector<std::shared_ptr<Sequence>> seqs = {seqPtr};
        RequestId thisRequestId = allocateTestData.requestId;
        SequenceGroupSPtr groupPtr = std::make_shared<SequenceGroup>(thisRequestId, seqs, sampling);
        seqGroupPtrs.push_back(groupPtr);
        seqPtrs.push_back(seqPtr);

        AllocStatus canAllocate = blockManager.CanAllocate(groupPtr);
        blockManager.Allocate(groupPtr);

        int numAllocatedBlocks = std::ceil(allocateTestData.inputs.size() * 1.0 / allocateTestData.blockSize);
        AllocatedNumBlocks.push_back(numAllocatedBlocks);
        numAllAllocatedBlocks += numAllocatedBlocks;
        currNumFreeNpuBlocks -= numAllocatedBlocks;
    }
    // 全部swapout
    for (int i = 0; i < sequencialAllocateTestData.size(); i++) {
        auto seqPtr = seqPtrs[i];
        auto groupPtr = seqGroupPtrs[i];
        // 设置序列状态为 RUNNING
        seqPtr->status_ = SequenceStatus::RUNNING;
        bool canSwapOut = blockManager.CanSwapOut(groupPtr);

        auto physicalBlockIdMapping = blockManager.SwapOut(groupPtr);
        currNumFreeCpuBlocks -= AllocatedNumBlocks[i];
        currNumFreeNpuBlocks += AllocatedNumBlocks[i];
    }
    // 再反向swapin
    for (int i = sequencialAllocateTestData.size() - 1; i >= 0; i--) {
        auto seqPtr = seqPtrs[i];
        auto groupPtr = seqGroupPtrs[i];
        // 设置序列状态为 SWAPPED
        seqPtr->status_ = SequenceStatus::SWAPPED;
        EXPECT_EQ(blockManager.GetNumFreeCpuBlocks(), currNumFreeCpuBlocks);
        EXPECT_EQ(blockManager.GetNumFreeNpuBlocks(), currNumFreeNpuBlocks);
        AllocStatus canSwapIn = blockManager.CanSwapIn(groupPtr, 0);

        auto physicalBlockIdMapping = blockManager.SwapIn(groupPtr);
        currNumFreeCpuBlocks += AllocatedNumBlocks[i];
        currNumFreeNpuBlocks -= AllocatedNumBlocks[i];

        EXPECT_EQ(physicalBlockIdMapping, sequencialSwapInTestData[i].physicalBlockIdMappingExpected);
        EXPECT_EQ(blockManager.GetNumFreeCpuBlocks(), currNumFreeCpuBlocks);
        EXPECT_EQ(blockManager.GetNumFreeNpuBlocks(), currNumFreeNpuBlocks);

        auto blockIds = blockManager.GetBlockIds(seqPtr->seqId_);
        ASSERT_EQ(blockIds.size(), 1u);
        EXPECT_EQ(blockIds[0], sequencialSwapInTestData[i].blockIdsExpectedAfterSwapIn);

        blockManager.Free(seqPtr->seqId_);
        currNumFreeNpuBlocks += AllocatedNumBlocks[i];
        EXPECT_EQ(blockManager.GetNumFreeCpuBlocks(), currNumFreeCpuBlocks);
        EXPECT_EQ(blockManager.GetNumFreeNpuBlocks(), currNumFreeNpuBlocks);
    }
    EXPECT_EQ(blockManager.GetNumFreeCpuBlocks(), cpuBlockNum);
    EXPECT_EQ(blockManager.GetNumFreeNpuBlocks(), npuBlockNum);
}

/**
 * Swap异常测试：不allocate直接swap，由于seqId2BlockTable_没有映射而抛出out_of_range异常
 */
TEST(SwapThrowTest, ExpectThrowWhenSwapWithoutAllocate)
{
    SequenceId seqId = 10;
    size_t blockSize = 8;
    size_t cpuBlockNum = 10;
    size_t npuBlockNum = 10;
    bool enableCaching = false;
    size_t reservedBlockNum = 0;
    size_t speculativeSlots = 0;
    std::shared_ptr<SamplingParams> sampling = nullptr;
    BlockManagerConfig config{blockSize,        cpuBlockNum,  npuBlockNum, reservedBlockNum,
                              speculativeSlots, enableCaching}; // blockSize=8, cpuBlockNum=10, npuBlockNum=10
    SelfAttnBlockManager blockManager(config);

    // 创建 Sequence 和 SequenceGroup
    SequenceSPtr seqPtr = std::make_shared<Sequence>(
        seqId, blockSize, std::vector<TokenId>{1, 2, 3, 4}); // seqId=10, blockSize=8,
    std::vector<std::shared_ptr<Sequence>> seqs = {seqPtr};
    RequestId requestId = "Req:0";
    SequenceGroupSPtr groupPtr = std::make_shared<SequenceGroup>(requestId, seqs, sampling);
    seqPtr->status_ = SequenceStatus::RUNNING;
    EXPECT_THROW({ blockManager.SwapOut(groupPtr); }, std::out_of_range);
    seqPtr->status_ = SequenceStatus::SWAPPED;
    EXPECT_THROW({ blockManager.SwapIn(groupPtr); }, std::out_of_range);
}
} // namespace mindie_llm
