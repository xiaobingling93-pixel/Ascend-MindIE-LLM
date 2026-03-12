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
#include <memory>
#include <unordered_set>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include <thread>
#define private public
#define protected public
#include "scheduler.h"
#include "config_info.h"
#include "sequence_group.h"
#include "sequence.h"
#include "sampling.h"
#include "block_manager_interface.h"
#include "self_attn_block_manager.h"
#include "concurrent_deque.h"
#include "layerwise_fcfs_policy.h"

using namespace mindie_llm;

static SequenceGroupSPtr createDummySeqGroup(std::string &requestId, int promptLength, int blockSize)
{
    std::vector<TokenId> dummyInputs;
    for (int i = 0; i < promptLength; ++i) {
        dummyInputs.push_back(i);
    }

    SequenceSPtr seq = std::make_shared<Sequence>(std::stol(requestId), blockSize, dummyInputs);
    auto samplingParams = std::make_shared<SamplingParams>();
    samplingParams->bestOf = 1;

    std::vector<SequenceSPtr> seqs{seq};
    SequenceGroupSPtr seqGroup = std::make_shared<SequenceGroup>(requestId, seqs, samplingParams);
    seqGroup->seqId2ParallelSeqGroup_.Insert(seqGroup->firstSeq->seqId_, seqGroup);
    seqGroup->parentSeqId_ = seqGroup->firstSeq->seqId_;
    return seqGroup;
}

class FcfsEdgeCloudTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        schedulerConfig_ = std::make_shared<SchedulerConfig>();
        schedulerConfig_->policyType = 0;
        schedulerConfig_->maxSeqLen = 200;
        schedulerConfig_->maxPrefillTokens = 200;
        schedulerConfig_->maxPrefillBatchSize = 4;
        schedulerConfig_->maxBatchSize = 4;
        schedulerConfig_->cacheBlockSize = blockSize;
        schedulerConfig_->cpuBlockNum = 300;
        schedulerConfig_->npuBlockNum = 300;
        schedulerConfig_->spSize = 1;
        blockConf_ = {schedulerConfig_->cacheBlockSize,
                     schedulerConfig_->cpuBlockNum,
                     schedulerConfig_->npuBlockNum,
                     0,
                     0,
                     schedulerConfig_->enablePrefixCache,
                     schedulerConfig_->spSize * schedulerConfig_->cpSize};
        blockManager_ = BlockManagerFactory::CreateBlockSpaceManager(BlockManagerType::SELFATTNBLOCKMANAGER,
                                                                    std::move(blockConf_), 0);
        policy_ = std::make_shared<LayerwiseFcfsPolicy>(schedulerConfig_, blockManager_);
    }
    int blockSize = 4;
    std::shared_ptr<SchedulerConfig> schedulerConfig_;
    std::shared_ptr<LayerwiseFcfsPolicy> policy_;
    BlockManagerConfig blockConf_;
    BlockSpaceManagerSPtr blockManager_;
};

TEST_F(FcfsEdgeCloudTest, TestApplyToRunningQueueEdgeCloud)
{
    int blockSize = 4;
    std::string reqId = "0";
    auto seqGroup = createDummySeqGroup(reqId, /*promptLength=*/blockSize, blockSize);
    policy_->queuesCollection_ = std::make_shared<SeqGroupCollection>();
    policy_->queuesCollection_->running_.push_front(seqGroup);
    SchedulingBudget budget(schedulerConfig_->maxPrefillTokens, schedulerConfig_->maxBatchSize, schedulerConfig_);
    RunningOutputs runningOutput = policy_->ApplyToRunningQueue(budget, false);
}

TEST_F(FcfsEdgeCloudTest, TestApplyToWaitingQueueEdgeCloud)
{
    int blockSize = 4;
    std::string reqId = "0";
    auto seqGroup = createDummySeqGroup(reqId, /*promptLength=*/blockSize, blockSize);
    policy_->queuesCollection_ = std::make_shared<SeqGroupCollection>();
    policy_->queuesCollection_->waiting_.push_front(seqGroup);
    SchedulingBudget budget(schedulerConfig_->maxPrefillTokens, schedulerConfig_->maxBatchSize, schedulerConfig_);
    PrefillOutputs runningOutput = policy_->ApplyToWaitingQueue(budget, false);
}