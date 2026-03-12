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
#include <deque>
#define private public // enable accessment to private memebers
#include "self_attn_block_manager.h"
#include "sequence_group.h"
#include "hashless_block_allocator.h"

namespace mindie_llm {

struct BlockManagerSetup {
    // for block manager
    size_t blockSize;
    size_t cpuBlockNum;
    size_t npuBlockNum;
    bool enableCaching;
    size_t numTests;
    BlockManagerSetup(size_t blkSize, size_t cpuBlkNum, size_t npuBlkNum, bool enblCaching, size_t numTsts)
        : blockSize(blkSize), cpuBlockNum(cpuBlkNum), npuBlockNum(npuBlkNum), enableCaching(enblCaching),
          numTests(numTsts)
    {
    }
};

struct OperationParams {
    RequestId requestId;
    std::string api;
    SequenceId seqId;
    std::vector<TokenId> prompts;
    std::vector<TokenId> tokensToAppend;
    std::vector<AllocStatus> canAllocate;
    std::vector<bool> canAppend;
    std::vector<bool> canSwapOut;
    std::vector<AllocStatus> canSwapIn;
    std::deque<BlockId> cpuFreeBlockIds;
    std::deque<BlockId> npuFreeBlockIds;
};

struct SystemTestData {
    BlockManagerSetup setup;
    OperationParams operations[100];
};

const SystemTestData systemTestData[] = {

    //-------------------------------- ST Scenario 1 --------------------------------
    // Scheduler View: Prefill => Recompute
    // BlockManager View: Allocate => Free

    // TestFlow/0 - Allocate with adequate NPU blocks
    {
        BlockManagerSetup(4, 1, 1, false, 2),
        OperationParams{"Req:0", "Allocate", 0, {1, 2, 3, 4}, {}, {AllocStatus::OK}, {}, {}, {}, {1}, {}},
        OperationParams{"Req:0", "Free", 0, {}, {}, {}, {}, {}, {}, {1}, {0}},
    },

    // TestFlow/1 - Allocate failed with inadequate NPU blocks
    {
        BlockManagerSetup(4, 1, 1, false, 4),
        OperationParams{"Req:0", "Allocate", 0, {1, 2, 3, 4, 5, 6}, {}, {AllocStatus::NEVER}, {}, {}, {}, {1}, {0}},
        OperationParams{"Req:1", "Allocate", 0, {1, 2, 3, 4}, {}, {AllocStatus::OK}, {}, {}, {}, {1}, {}},
        OperationParams{"Req:2", "Allocate", 0, {6, 7, 8, 9}, {}, {AllocStatus::LATER}, {}, {}, {}, {1}, {}},
        OperationParams{"Req:0", "Free", 0, {}, {}, {}, {}, {}, {}, {1}, {0}},
    },

    //-------------------------------- ST Scenario 2 --------------------------------
    // Scheduler View: Prefill => Decode => Recompute/End
    // BlockManager View: Allocate => AppendSlot => Free

    // TestFlow/2 - AppendSlot with adequate NPU blocks
    {
        BlockManagerSetup(4, 2, 2, false, 3),
        OperationParams{"Req:0", "Allocate", 0, {1, 2, 3, 4}, {}, {AllocStatus::OK}, {}, {}, {}, {2, 3}, {1}},
        OperationParams{"Req:0", "AppendSlot", 0, {}, {5}, {}, {true}, {}, {}, {2, 3}, {}},
        OperationParams{"Req:0", "Free", 0, {}, {}, {}, {}, {}, {}, {2, 3}, {0, 1}},
    },

    // TestFlow/3 - ApppendSlot failed with inadequate NPU blocks
    {
        BlockManagerSetup(4, 2, 1, false, 2),
        OperationParams{"Req:0", "Allocate", 0, {1, 2, 3, 4}, {}, {AllocStatus::OK}, {}, {}, {}, {1, 2}, {}},
        OperationParams{"Req:0", "AppendSlot", 0, {}, {5}, {}, {false}, {}, {}, {1, 2}, {}},
    },

    //-------------------------------- ST Scenario 3 --------------------------------
    // Scheduler View: Prefill => Decode => SwapOut => SwapIn => Recompute/End
    // BlockManager View: Allocate => AppendSlot => SwapOut => SwapIn => Free

    // TestFlow/4 - SwapOut/SwapIn with adequate CPU/NPU resources
    {BlockManagerSetup(4, 2, 2, false, 5),
     OperationParams{"Req:0", "Allocate", 0, {1, 2, 3, 4}, {}, {AllocStatus::OK}, {}, {}, {}, {2, 3}, {1}},
     OperationParams{"Req:0", "AppendSlot", 0, {}, {5}, {}, {true}, {}, {}, {2, 3}, {}},
     OperationParams{"Req:0", "SwapOut", 0, {}, {}, {}, {}, {true}, {}, {}, {0, 1}},
     OperationParams{"Req:0", "SwapIn", 0, {}, {}, {}, {}, {}, {AllocStatus::OK}, {2, 3}, {}},
     OperationParams{"Req:0", "Free", 0, {}, {}, {}, {}, {}, {}, {2, 3}, {0, 1}}},

    // TestFlow/5 - SwapOut failed with inadequate CPU resources
    {BlockManagerSetup(4, 0, 2, false, 3),
     OperationParams{"Req:0", "Allocate", 0, {1, 2, 3, 4}, {}, {AllocStatus::OK}, {}, {}, {}, {}, {1}},
     OperationParams{"Req:0", "AppendSlot", 0, {}, {5}, {}, {true}, {}, {}, {}, {}},
     OperationParams{"Req:0", "SwapOut", 0, {}, {}, {}, {}, {false}, {}, {}, {}}},

    // TestFlow/6 - SwapIn failed with inadequate NPU resources
    {
        // intuitively npuBlockNum should be 1 to simulate SwapIn failure with inadequate NPU resources
        // however, we need to set it to 1+1=2 since there is a redundancy design for CoW
        // check for details: self_attn_block_manager.cpp:95
        BlockManagerSetup(4, 1, 2, false, 6),
        OperationParams{"Req:0", "Allocate", 0, {1, 2, 3}, {}, {AllocStatus::OK}, {}, {}, {}, {2}, {1}},
        OperationParams{"Req:0", "AppendSlot", 0, {}, {4}, {}, {true}, {}, {}, {2}, {1}},
        OperationParams{"Req:0", "SwapOut", 0, {}, {}, {}, {}, {true}, {}, {}, {1, 0}},
        OperationParams{"Req:1", "Allocate", 1, {5, 6, 7}, {}, {AllocStatus::OK}, {}, {}, {}, {}, {0}},
        OperationParams{"Req:2", "Allocate", 2, {8, 9, 10}, {}, {AllocStatus::OK}, {}, {}, {}, {}, {}},
        OperationParams{"Req:0", "SwapIn", 0, {}, {}, {}, {}, {}, {AllocStatus::LATER}, {}, {}},
    },

    //-------------------------------- ST Scenario 4 --------------------------------
    // Scehduler View: Prefill => SwapOut => SwapIn => Recompute/End
    // BlockManager View: Allocate => SwapOut => SwapIn => Free

    // TestFlow/7 - SwapOut with adequate CPU blocks (single-block scenario) @ jkq swapOutTestData 1
    {
        BlockManagerSetup(8, 2, 2, false, 3),
        OperationParams{
            "Req:0", "Allocate", 0, {1, 2, 3, 4, 5, 6, 7, 8}, {}, {AllocStatus::OK}, {}, {}, {}, {2, 3}, {1}},
        OperationParams{"Req:0", "SwapOut", 0, {}, {}, {}, {}, {true}, {}, {3}, {1, 0}},
        OperationParams{"Req:0", "SwapIn", 0, {}, {}, {}, {}, {}, {AllocStatus::OK}, {3, 2}, {0}},
        OperationParams{"Req:0", "Free", 0, {}, {}, {}, {}, {}, {}, {3, 2}, {0, 1}},
    },

    // TestFlow/8 - SwapOut with adequate CPU blocks (multi-block scenario) @ jkq swapOutTestData 2
    {BlockManagerSetup(4, 2, 2, false, 4),
     OperationParams{"Req:0", "Allocate", 0, {1, 2, 3, 4, 5, 6, 7, 8}, {}, {AllocStatus::OK}, {}, {}, {}, {2, 3}, {}},
     OperationParams{"Req:0", "SwapOut", 0, {}, {}, {}, {}, {true}, {}, {}, {0, 1}},
     OperationParams{"Req:0", "SwapIn", 0, {}, {}, {}, {}, {}, {AllocStatus::OK}, {2, 3}, {}},
     OperationParams{"Req:0", "Free", 0, {}, {}, {}, {}, {}, {}, {2, 3}, {0, 1}}},

    // TestFlow/9 - SwapOut failed with inadequate CPU blocks (single-block scenario) @ jkq swapOutTestData 3
    {
        BlockManagerSetup(2, 1, 2, false, 2),
        OperationParams{"Req:0", "Allocate", 0, {1, 2, 3, 4}, {}, {AllocStatus::OK}, {}, {}, {}, {2}, {}},
        OperationParams{"Req:0", "SwapOut", 0, {}, {}, {}, {}, {false}, {}, {2}, {}},
    },

    //-------------------------------- ST Scenario 5 --------------------------------
    // Scehduler View: Prefill => SwapOut => SwapIn => Decode => Recompute/End
    // BlockManager View: Allocate => SwapOut => SwapIn => AppendSlot => Free

    // TestFlow/10 - Append with adequate NPU blocks
    {
        BlockManagerSetup(4, 2, 3, false, 5),
        OperationParams{"Req:0", "Allocate", 0, {1, 2, 3, 4, 5, 6}, {}, {AllocStatus::OK}, {}, {}, {}, {3, 4}, {2}},
        OperationParams{"Req:0", "SwapOut", 0, {}, {}, {}, {}, {true}, {}, {}, {2, 0, 1}},
        OperationParams{"Req:0", "SwapIn", 0, {}, {}, {}, {}, {}, {AllocStatus::OK}, {3, 4}, {1}},
        OperationParams{"Req:0", "AppendSlot", 0, {}, {7, 8}, {}, {true}, {}, {}, {3, 4}, {1}},
        OperationParams{"Req:0", "Free", 0, {}, {}, {}, {}, {}, {}, {3, 4}, {1, 2, 0}},
    },

    // TestFlow/11 - Append with inadequate NPU blocks
    {
        BlockManagerSetup(4, 2, 3, false, 4),
        OperationParams{"Req:0", "Allocate", 0, {1, 2, 3, 4, 5, 6}, {}, {AllocStatus::OK}, {}, {}, {}, {3, 4}, {2}},
        OperationParams{"Req:0", "SwapOut", 0, {}, {}, {}, {}, {true}, {}, {}, {2, 0, 1}},
        OperationParams{"Req:0", "SwapIn", 0, {}, {}, {}, {}, {}, {AllocStatus::OK}, {3, 4}, {1}},
        OperationParams{"Req:0", "AppendSlot", 0, {}, {7, 8, 9, 10, 11, 12}, {}, {true}, {}, {}, {3, 4}, {}},
    },

    //-------------------------------- ST Scenario 6 --------------------------------
    // Scheduler View: Prefill => Decode => SwapOut => SwapIn => Decode => Recompute/End
    // BlockManager View: Allocate => AppendSlot => SwapOut => SwapIn => AppendSlot => Free

    // TestFlow/12 - Append with adequate NPU blocks
    {
        BlockManagerSetup(4, 2, 4, false, 6),
        OperationParams{"Req:0", "Allocate", 0, {1, 2, 3, 4, 5, 6}, {}, {AllocStatus::OK}, {}, {}, {}, {4, 5}, {2, 3}},
        OperationParams{"Req:0", "AppendSlot", 0, {}, {7, 8}, {}, {true}, {}, {}, {4, 5}, {2, 3}},
        OperationParams{"Req:0", "SwapOut", 0, {}, {}, {}, {}, {true}, {}, {}, {2, 3, 0, 1}},
        OperationParams{"Req:0", "SwapIn", 0, {}, {}, {}, {}, {}, {AllocStatus::OK}, {4, 5}, {0, 1}},
        OperationParams{"Req:0", "AppendSlot", 0, {}, {9, 10}, {}, {true}, {}, {}, {4, 5}, {1}},
        OperationParams{"Req:0", "Free", 0, {}, {}, {}, {}, {}, {}, {4, 5}, {1, 2, 3, 0}},
    },

    // TestFlow/13 - Append with inadequate NPU blocks
    {
        BlockManagerSetup(4, 2, 3, false, 6),
        OperationParams{"Req:0", "Allocate", 0, {1, 2, 3, 4, 5, 6}, {}, {AllocStatus::OK}, {}, {}, {}, {3, 4}, {2}},
        OperationParams{"Req:0", "AppendSlot", 0, {}, {7, 8}, {}, {true}, {}, {}, {3, 4}, {2}},
        OperationParams{"Req:0", "SwapOut", 0, {}, {}, {}, {}, {true}, {}, {}, {2, 0, 1}},
        OperationParams{"Req:0", "SwapIn", 0, {}, {}, {}, {}, {}, {AllocStatus::OK}, {3, 4}, {1}},
        OperationParams{"Req:1", "Allocate", 1, {9, 10, 11, 12}, {}, {AllocStatus::OK}, {}, {}, {}, {3, 4}, {}},
        OperationParams{"Req:0", "AppendSlot", 0, {}, {13, 14}, {}, {false}, {}, {}, {3, 4}, {}},
    },
};

class BlockManagerSystemTest : public ::testing::TestWithParam<SystemTestData> {
protected:
    // for sequence group
    std::shared_ptr<SamplingParams> sampling;
    // for block manager
    size_t reservedBlockNum = 0;
    size_t speculativeSlots = 0;
    // for utils
    BlockTable blockTable;
};

std::vector<BlockId> GetAllBlockIds(BlockSpaceManagerSPtr blockManagerPtr, DeviceType deviceType)
{
    // get all block ids from (the allocator of ) given device type
    std::shared_ptr<HashLessBlockAllocator> hashlessBlockAllocator = std::dynamic_pointer_cast<HashLessBlockAllocator>(
        std::dynamic_pointer_cast<CpuNpuBlockAllocator>(
            std::dynamic_pointer_cast<SelfAttnBlockManager>(blockManagerPtr)->blockAllocator_)
            ->GetAllocator(deviceType)); // pointer convert
    return hashlessBlockAllocator->allBlockIndices_;
}

std::deque<BlockId> GetFreeBlockIds(BlockSpaceManagerSPtr blockManagerPtr, DeviceType deviceType)
{
    // get free block ids from (the allocator of ) given device type
    std::shared_ptr<HashLessBlockAllocator> hashlessBlockAllocator = std::dynamic_pointer_cast<HashLessBlockAllocator>(
        std::dynamic_pointer_cast<CpuNpuBlockAllocator>(
            std::dynamic_pointer_cast<SelfAttnBlockManager>(blockManagerPtr)->blockAllocator_)
            ->GetAllocator(deviceType)); // pointer convert
    return hashlessBlockAllocator->freeBlockIndices_;
}

/**/
TEST_P(BlockManagerSystemTest, TestFlow)
{
    //-------------------------------------------prepare-------------------------------------------

    // init
    SystemTestData stData = GetParam();
    BlockManagerConfig config{stData.setup.blockSize, stData.setup.cpuBlockNum, stData.setup.npuBlockNum,
                              reservedBlockNum,       speculativeSlots,         stData.setup.enableCaching};
    SelfAttnBlockManager blockManager(config);
    BlockSpaceManagerSPtr blockManagerPtr = std::make_shared<SelfAttnBlockManager>(blockManager);

    // declare variables
    std::vector<BlockId> cpuAllBlockIds, npuAllBlockIds;                     // GET: CPU/NPU block ids
    std::deque<BlockId> cpuMockFreeBlockIds, npuMockFreeBlockIds;            // MOCK: CPU/NPU block ids
    size_t numCpuBlocksBefore, numNpuBlocksBefore;                           // GET: block num before operation
    size_t numCpuBlocksAfter, numNpuBlocksAfter;                             // GET: block num after operation
    size_t numCpuChangeThisTime, numNpuChangeThisTime;                       // GET: block num used for this operation
    std::vector<BlockId> cpuMockAllocatedThisTime, npuMockAllocatedThisTime; // MOCK: allocated after 1 operation
    AllocStatus canAllocate;
    bool canAppend, canSwapOut;
    std::vector<std::vector<TokenId>> tokensChunked, outputsChunked;
    std::vector<BlockId> cpuAllocatedBlockIds, npuAllocatedBlockIds;
    std::vector<std::pair<PhysicalBlockId, PhysicalBlockId>> physicalBlockIdMapping;
    std::vector<std::pair<BlockId, BlockId>> allCowAfterAppend;

    // CPU all / mock ids
    cpuAllBlockIds = GetAllBlockIds(blockManagerPtr, DeviceType::CPU);
    // NPU all / mock ids
    npuAllBlockIds = GetAllBlockIds(blockManagerPtr, DeviceType::NPU);

    // sequences & sequence groups
    std::unordered_map<RequestId, SequenceGroupSPtr> reqid2ptr;
    // Sequence seq;
    SequenceSPtr seqPtr, thisSeqPtr;
    // SequenceGroup seqGroup;
    SequenceGroupSPtr groupPtr;

    //-------------------------------------------operate-------------------------------------------

    for (auto i = 0; i < stData.setup.numTests; i++) {

        auto thisOperation = stData.operations[i];

        // create SequenceGroup
        auto it = reqid2ptr.find(thisOperation.requestId);
        if (it == reqid2ptr.end()) { // does not exist
            // create a new sequence
            seqPtr = std::make_shared<Sequence>(
                Sequence(thisOperation.seqId, stData.setup.blockSize, thisOperation.prompts));
            // create a new sequence group
            reqid2ptr[thisOperation.requestId] = std::make_shared<SequenceGroup>(
                thisOperation.requestId, std::vector<std::shared_ptr<Sequence>>({seqPtr}), sampling);
        }
        groupPtr = reqid2ptr[thisOperation.requestId];
        thisSeqPtr = groupPtr->firstSeq;

        // before operation

        if (thisOperation.api == "Allocate") {

            // allocate (Prefill)
            canAllocate = blockManager.CanAllocate(groupPtr);
            EXPECT_EQ(canAllocate, thisOperation.canAllocate[0]); // CHECKPOINT 1: CanAllocate
            if (!(canAllocate == AllocStatus::OK)) {              // allocate failed
                continue;
            }
            EXPECT_EQ(blockManager.Allocate(groupPtr), true); // CHECKPOINT 2: Allocate
            thisSeqPtr->status_ = SequenceStatus::RUNNING;
        }

        if (thisOperation.api == "AppendSlot") {

            // append (Decode)
            thisSeqPtr->data_.outputTokenIds.insert(thisSeqPtr->data_.outputTokenIds.end(),
                                                    thisOperation.tokensToAppend.begin(),
                                                    thisOperation.tokensToAppend.end()); // MOCK: decode
            canAppend = blockManager.CanAppendSlot(groupPtr);
            EXPECT_EQ(canAppend, thisOperation.canAppend[0]); // CHECKPOINT 1: CanAppendSlots
            if (!canAppend) {
                continue;
            }
            EXPECT_EQ(thisSeqPtr->status_, SequenceStatus::RUNNING); // CHECKPOINT 2: Sequence.status_
            allCowAfterAppend = blockManager.AppendSlot(thisSeqPtr);
        }

        if (thisOperation.api == "SwapOut") {

            // swap out
            canSwapOut = blockManager.CanSwapOut(groupPtr); // CHECKPOINT 1: CanSwapOut
            EXPECT_EQ(canSwapOut, thisOperation.canSwapOut[0]);
            if (!(canSwapOut == true)) { // swap-out failed
                continue;
            }
            EXPECT_EQ(thisSeqPtr->status_, SequenceStatus::RUNNING);
            physicalBlockIdMapping = blockManager.SwapOut(groupPtr); // CHECKPOINT 2: SwapOut
            thisSeqPtr->status_ = SequenceStatus::SWAPPED;
        }

        if (thisOperation.api == "SwapIn") {

            // swap in
            AllocStatus canSwapIn = blockManager.CanSwapIn(groupPtr, 0); // CHECKPOINT 1: CanSwapIn, 0 for lookahead
            EXPECT_EQ(canSwapIn, thisOperation.canSwapIn[0]);
            if (!(canSwapIn == AllocStatus::OK)) { // swap-in failed
                continue;
            }
            EXPECT_EQ(groupPtr->firstSeq->status_, SequenceStatus::SWAPPED);
            physicalBlockIdMapping = blockManager.SwapIn(groupPtr); // CHECKPOINT 2: SwapIn
            thisSeqPtr->status_ = SequenceStatus::RUNNING;
        }

        if (thisOperation.api == "Free") {

            // free (End / Recompute)
            EXPECT_NO_THROW(blockManager.Free(thisOperation.seqId)); // CHECKPOINT 2: Free
            thisSeqPtr->status_ = SequenceStatus::WAITING;
        }

        // CHECKPOINT 3: free block ids after operation
        EXPECT_EQ(GetFreeBlockIds(blockManagerPtr, DeviceType::CPU), thisOperation.cpuFreeBlockIds);
        EXPECT_EQ(GetFreeBlockIds(blockManagerPtr, DeviceType::NPU), thisOperation.npuFreeBlockIds);
        // CHECKPOINT 4: used block ids
        auto index = blockManager.seqId2BlockTable_.find(thisOperation.seqId);
        if (index != blockManager.seqId2BlockTable_.end()) { // sequence still exists
            const auto allBlockIds = blockManager.GetBlockIds(thisOperation.seqId);
            ASSERT_EQ(allBlockIds.size(), 1u);
            EXPECT_EQ(allBlockIds[0].size(),
                      blockTable.ChunkTokensForAllocate(thisSeqPtr->GetTokenIds(), stData.setup.blockSize).size());
        }
    }
};
INSTANTIATE_TEST_SUITE_P(TestFlow, BlockManagerSystemTest, ::testing::ValuesIn(systemTestData));

} // namespace mindie_llm
