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
#include "model_exec_output_handler.h"
#include "live_infer_context.h"
#include "policy/stage_policy/stage_policy.h"

using namespace mindie_llm;

class ModelExecOutputHandlerTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        SchedulerConfigSPtr schedulerConfigPtr = std::make_shared<SchedulerConfig>();
        schedulerConfigPtr->bufferResponseEnabled = false;
        schedulerConfigPtr->prefillExpectedTime = 0;
        schedulerConfigPtr->decodeExpectedTime = 0;
        
        auto predictor = std::make_shared<LatencyPredictor>();
        modelExecOutputHandler_ = std::make_shared<ModelExecOutputHandler>(
            [this](ResponseSPtr response) { responses_.push_back(response); }, Role::PnD,
            schedulerConfigPtr, predictor);
    }
    
    class TestStagePolicy : public StagePolicy {
    public:
        mutable bool markInferenceEndTimeStampCalled = false;
        
        PDPriorityType Apply(ConcurrentDeque<SequenceGroupSPtr> &,
                           ConcurrentDeque<SequenceGroupSPtr> &,
                           ConcurrentDeque<SequenceGroupSPtr> &) override
        {
            return PDPriorityType::PREFILL_FIRST;
        }
        
        void MarkInferenceEndTimeStamp() override
        {
            markInferenceEndTimeStampCalled = true;
        }
    };
    
    static ModelBatchResultSPtr CreateModelBatchResult(int64_t startSeqId)
    {
        model_execute_data::ExecuteModelResponse modelBatchResult;
        for (int i = 0; i < 3; i++) { // request_num = 3
            model_execute_data::CompletionSequenceGroupOutput *outputs = modelBatchResult.add_outputs();
            model_execute_data::SequenceOutput *samples = outputs->add_samples();
            samples->set_seq_id(startSeqId + i);
            samples->set_parent_seq_id(startSeqId);
            samples->set_finish_reason(i); // 0: continue, 1: eos, else exception
            samples->set_num_speculative_tokens(1);
            samples->set_truncation_index(-10);
            samples->set_cumulative_logprobs(0.6);
            samples->set_num_parallel_tokens(1);
            samples->add_output_token(100);
            samples->add_logprob(0.9);
            samples->add_top_token_ids(100);
            samples->add_top_token_ids(101);
            samples->add_top_logprobs(0.6);
            samples->add_top_logprobs(0.5);
            SequenceSPtr seqSPtr = std::make_shared<Sequence>(startSeqId + i, 0, std::vector<TokenId>{100});
            std::vector<SequenceSPtr> seqs = {seqSPtr};
            RequestId reqId = std::to_string(startSeqId + i);
            SamplingParamsSPtr sampling = std::make_shared<SamplingParams>();
            SequenceGroupSPtr seqGrpSPtr = std::make_shared<SequenceGroup>(reqId, seqs, sampling);
            seqGrpSPtr->metrics_.inferReqId_ = std::to_string(startSeqId + i);
            seqGrpSPtr->pInstanceId = 0;
            seqGrpSPtr->pBlockTable = std::vector<BlockIds>{
                BlockIds{static_cast<BlockId>(i + 1), static_cast<BlockId>(i + 2)}};

            LiveInferContext::GetInstance(0)->Add(seqGrpSPtr);
        }

        return std::make_shared<model_execute_data::ExecuteModelResponse>(modelBatchResult);
    }

    std::shared_ptr<ModelExecOutputHandler> modelExecOutputHandler_;
    std::vector<ResponseSPtr> responses_;
};

TEST_F(ModelExecOutputHandlerTest, ShouldNotUpdateWhenSeqIdNotInMap)
{
    // seqId2RequestIdMap_ 为空，asyncBatchNum_ 不为 0
    modelExecOutputHandler_->GetAsyncBatchNum().store(2);
    ModelBatchResultSPtr modelBatchResult = CreateModelBatchResult(100);
    modelExecOutputHandler_->Entry4Executor(modelBatchResult);
    EXPECT_EQ(modelExecOutputHandler_->GetAsyncBatchNum(), 1);
    EXPECT_EQ(modelExecOutputHandler_->GetFinishedSeqIds().Size(), 1);
    EXPECT_EQ(modelExecOutputHandler_->GetExceptionSeqIds().Size(), 1);
    EXPECT_EQ(modelExecOutputHandler_->GetSeqIdToOutputTokenQueue().Size(), 3);
}

TEST_F(ModelExecOutputHandlerTest, ShouldUpdateWhenSeqIdInMap)
{
    ModelBatchResultSPtr modelBatchResult = CreateModelBatchResult(200);
    // 模拟 engine 设置 seqId2RequestIdMap_, asyncBatchNum_
    modelExecOutputHandler_->GetAsyncBatchNum().store(2);
    
    EXPECT_EQ(modelExecOutputHandler_->stagePolicy_, nullptr);
    
    modelExecOutputHandler_->Entry4Executor(modelBatchResult);
    EXPECT_EQ(modelExecOutputHandler_->GetAsyncBatchNum(), 1);

    // finishedSeqIds_ 应为 {2}
    size_t expectedFinishedSeqId = 201;
    ConcurrentDeque<SequenceId> finishedSeqIds = modelExecOutputHandler_->GetFinishedSeqIds();
    while (!finishedSeqIds.Empty()) { // 保证deque里没有多余的seqId
        SequenceId finishedSeqId;
        finishedSeqIds.PopFront(finishedSeqId);
        EXPECT_EQ(finishedSeqId, expectedFinishedSeqId);
    }
    // execExceptionSeqIds_ 应为 {3}
    size_t expectedExceptionSeqId = 202;
    ConcurrentDeque<SequenceId> exceptionSeqIds = modelExecOutputHandler_->GetExceptionSeqIds();
    while (!exceptionSeqIds.Empty()) {
        SequenceId exceptionSeqId;
        exceptionSeqIds.PopFront(exceptionSeqId);
        EXPECT_EQ(exceptionSeqId, expectedExceptionSeqId);
    }
    // seqIdToOutputTokenQueue_ 应为 {1, 100}, {2, 100}, {3, 100}
    std::pair<SequenceId, TokenId> expectedSeqIdToToken = std::pair{200, 100};
    ConcurrentDeque<std::pair<SequenceId, TokenId>> seqIdToOutputTokenQueue =
        modelExecOutputHandler_->GetSeqIdToOutputTokenQueue();
    while (!seqIdToOutputTokenQueue.Empty()) {
        std::pair<SequenceId, TokenId> SeqIdToToken;
        seqIdToOutputTokenQueue.PopFront(SeqIdToToken);
        EXPECT_EQ(SeqIdToToken, expectedSeqIdToToken);
        expectedSeqIdToToken.first += 1;
    }
}



TEST_F(ModelExecOutputHandlerTest, TestCollectTensorData)
{
    ModelBatchResultSPtr modelBatchResult = CreateModelBatchResult(300);
    
    // Test all 3 samples
    for (int i = 0; i < 3; i++) {
        model_execute_data::CompletionSequenceGroupOutput output = modelBatchResult->outputs(i);
        // Create a valid InferRequestId for the test
        RequestIdNew reqId(std::to_string(300 + i));
        ResponseSPtr response = std::make_shared<Response>(reqId);
        modelExecOutputHandler_->AddOutputsToResponse(response, output);
        
        EXPECT_EQ(response->responseContents.size(), 1);
        const ResponseContent& responseContent = response->responseContents[0];
        
        // verify basic fields
        EXPECT_EQ(responseContent.seqId, 300 + i);
        EXPECT_EQ(responseContent.parentSeqId, 300);
        EXPECT_EQ(static_cast<int>(responseContent.finishReason), i);
        EXPECT_EQ(responseContent.speculativeTokenNum, 1);
        EXPECT_EQ(responseContent.cumLogProb, 0.6f);
        EXPECT_EQ(responseContent.truncationIndex, -10);
        
        // verify token and probability vectors
        EXPECT_EQ(responseContent.outTokenIds, std::vector<TokenId>({100}));
        EXPECT_EQ(responseContent.outLogProbs, std::vector<float>({0.9f}));
        EXPECT_EQ(responseContent.topLogProbTokenIds, std::vector<TokenId>({100, 101}));
        EXPECT_EQ(responseContent.topLogProbs, std::vector<float>({0.6f, 0.5f}));
        
        // verify default fields
        EXPECT_TRUE(responseContent.srcBlockTable.empty());
        EXPECT_EQ(responseContent.singleLLMPrefillReqHandlerId, 0);
        EXPECT_EQ(responseContent.pdErrorCode, 0);
    }
}

TEST_F(ModelExecOutputHandlerTest, TestConvertSequenceGroupOutputToResponse)
{
    ModelBatchResultSPtr modelBatchResult = CreateModelBatchResult(400);
    for (int i = 0; i < 3; i++) {
        model_execute_data::CompletionSequenceGroupOutput output = modelBatchResult->outputs(i);
        RequestIdNew expectedReqId = std::to_string(400 + i);
        uint64_t queueWaitTime = 0;
        uint64_t currentPrefixCachedTokenNums = 0;
        ResponseSPtr response = modelExecOutputHandler_->ConvertSequenceGroupOutputToResponse(output,
            queueWaitTime, currentPrefixCachedTokenNums);
        if (i == 0) {
            EXPECT_FALSE(response->isEos);
        } else {
            EXPECT_TRUE(response->isEos);
        }
        EXPECT_EQ(static_cast<int>(response->inferStatusFlag), i);
        EXPECT_EQ(response->reqId, expectedReqId);
        // check response contents
        EXPECT_EQ(response->responseContents[0].seqId, 400+i);
        EXPECT_EQ(response->responseContents[0].parentSeqId, 400);
        EXPECT_EQ(response->responseContents[0].outTokenIds, std::vector<TokenId>({100}));
        EXPECT_EQ(response->responseContents[0].outLogProbs, std::vector<float>({0.9}));
        EXPECT_EQ(static_cast<int>(response->responseContents[0].finishReason), i);
        EXPECT_EQ(response->responseContents[0].speculativeTokenNum, 1);
        EXPECT_EQ(response->responseContents[0].truncationIndex, -10);
        EXPECT_EQ(response->responseContents[0].topLogProbTokenIds, std::vector<TokenId>({100, 101}));
        EXPECT_EQ(response->responseContents[0].topLogProbs, std::vector<float>({0.6, 0.5}));
        EXPECT_FLOAT_EQ(response->responseContents[0].cumLogProb, 0.6);
    }
}

TEST_F(ModelExecOutputHandlerTest, ShouldRemovePrefilledResponseWhenPublishKvCache)
{
    ModelBatchResultSPtr modelBatchResult = CreateModelBatchResult(500);
    
    auto testStagePolicy = std::make_shared<TestStagePolicy>();
    modelExecOutputHandler_->SetStagePolicy(testStagePolicy);
    
    modelExecOutputHandler_->GetAsyncBatchNum().store(1);
    
    EXPECT_FALSE(testStagePolicy->markInferenceEndTimeStampCalled);
    
    modelExecOutputHandler_->Entry4Executor(modelBatchResult);
    
    EXPECT_TRUE(testStagePolicy->markInferenceEndTimeStampCalled);
    
}

TEST_F(ModelExecOutputHandlerTest, ShouldReturnCorrectTensorWhenPublishKvCache)
{
    modelExecOutputHandler_->SetRole(Role::P);
    ModelBatchResultSPtr modelBatchResult = CreateModelBatchResult(600);
    
    modelBatchResult->mutable_outputs(0)->mutable_samples(0)->set_finish_reason(0); // INFER_RESPONSE_CONTINUE = 0
    modelExecOutputHandler_->GetAsyncBatchNum().store(1);
    size_t initialResponseCount = responses_.size();
    
    modelExecOutputHandler_->AsyncPublishPrefilledKvCache(modelBatchResult);
    EXPECT_GT(responses_.size(), initialResponseCount); // AsyncPublishPrefilledKvCache only adds response when finish_reason = 0
    ResponseSPtr response = responses_.back();
    EXPECT_EQ(response->transferStatusFlag, TransferStatusType::PUBLISH_KV_COMPLETE);
    // check response contents
    EXPECT_EQ(response->responseContents[0].srcBlockTable, std::vector<std::vector<int64_t>>({{1, 2}}));
    EXPECT_EQ(response->responseContents[0].singleLLMPrefillReqHandlerId, 0);
    
    size_t beforeEntry4ExecutorCount = responses_.size();
    modelExecOutputHandler_->Entry4Executor(modelBatchResult);
    EXPECT_GT(responses_.size(), beforeEntry4ExecutorCount);
}