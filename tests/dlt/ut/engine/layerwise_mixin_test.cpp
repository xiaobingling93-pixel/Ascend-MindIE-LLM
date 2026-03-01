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
#define private public
#include "layerwise_mixin/layerwise_mixin.h"

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

class LayerwiseMixinTest : public ::testing::Test {
protected:
    void SetUp() override
    {
    }

    LayerwiseMixin layerwiseMixin_;
};

TEST_F(LayerwiseMixinTest, LwdPrepareBatchTest)
{
    SchedulerOutputs out;
    out.forwardMode_ = ForwardMode::PREFILL;
    int blockSize = 4;
    std::string reqId = "0";
    auto seqGroup = createDummySeqGroup(reqId, /*promptLength=*/blockSize, blockSize);
    auto prefillSeqGrp = std::make_shared<ScheduledSequenceGroup>(seqGroup, blockSize);
    auto seq = seqGroup->firstSeq;
    seq->data_.layerwiseRecompute_ = true;
    seq->data_.layerwiseRecomputeReturn_ = true;
    seq->data_.layerwiseRunning_ = false;
    out.scheduledSeqGroups_.push_back(prefillSeqGrp);
    layerwiseMixin_.LwdPrepareBatch(true, out);
    EXPECT_TRUE(seq->data_.layerwiseRunning_);
    EXPECT_FALSE(seq->data_.layerwiseRecompute_);
    EXPECT_FALSE(seq->data_.layerwiseRecomputeReturn_);
}

TEST_F(LayerwiseMixinTest, LwdProcessResponseTest)
{
    int blockSize = 4;
    std::string reqId = "0";
    auto seqGroup = createDummySeqGroup(reqId, /*promptLength=*/blockSize, blockSize);
    auto seq = seqGroup->firstSeq;
    seq->data_.layerwiseStage_ = SequenceStage::PREFILL;
    seq->data_.layerwiseRecompute_ = true;
    seq->data_.layerwiseRecomputeReturn_ = false;
    seq->data_.layerwiseRunning_ = true;
    ForwardMode lastForwardMode = ForwardMode::PREFILL;
    std::deque<SequenceGroupSPtr> recomputeInDBatchQueue;
    EXPECT_TRUE(layerwiseMixin_.LwdProcessResponse(true, seqGroup, lastForwardMode, ForwardMode::PREFILL, recomputeInDBatchQueue));
    EXPECT_TRUE(seq->data_.layerwiseRecomputeReturn_);
}