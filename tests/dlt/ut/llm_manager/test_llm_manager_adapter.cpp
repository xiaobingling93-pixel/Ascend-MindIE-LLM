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
 
#include "test_llm_manager_adapter.h"
#include "test_llm_manager_response_spy.h"
#include "test_llm_manager_response_stub.h"

namespace mindie_llm {
// 检查v1 InferRequest中的INPUT_IDS tensor正确转换为v2的struct Request的input_ids字段
TEST_F(LlmManagerTest, should_return_v2_input_ids_ok_when_input_v1_request_ids_tensor)
{
    // given
    std::map<std::string, std::string> ipInfo = {{"infer_mode", "dmi"}};
    std::shared_ptr<LlmManager> llmManager = std::make_shared<LlmManager>("configPath_", GetRequestsStubInputIds, nullptr, nullptr, nullptr, nullptr, ipInfo);
    std::vector<int64_t> inferTokens = {1, 2, 3};
    
    // when
    auto requests = llmManager->impl_->getRequests_();

    // then
    ASSERT_EQ(requests.size(), 1);
    ASSERT_EQ(requests[0]->requestId, "req_1");
    ASSERT_EQ(requests[0]->input_token_num, 3);
    ASSERT_EQ(requests[0]->input_ids, inferTokens);
}

TEST_F(LlmManagerTest, should_return_v2_lora_id_ok_when_input_v1_lora_id_tensor)
{
    // given
    std::map<std::string, std::string> ipInfo = {{"infer_mode", "dmi"}};
    std::shared_ptr<LlmManager> llmManager = std::make_shared<LlmManager>("configPath_", GetRequestsStubLoraId, nullptr, nullptr, nullptr, nullptr, ipInfo);

    std::string loraId = "test_lora_123";
    
    // when
    auto requests = llmManager->impl_->getRequests_();

    // then
    ASSERT_EQ(requests.size(), 1);
    ASSERT_EQ(requests[0]->requestId, "req_1");
    ASSERT_EQ(requests[0]->loraId, loraId);
}

TEST_F(LlmManagerTest, should_return_v2_ignore_eos_ok_when_input_v1_ignore_eos_tensor)
{
    // given
    std::map<std::string, std::string> ipInfo = {{"infer_mode", "dmi"}};
    std::shared_ptr<LlmManager> llmManager = std::make_shared<LlmManager>("configPath_", GetRequestsStubIgnoreEos, nullptr, nullptr, nullptr, nullptr, ipInfo);

    bool ignoreEos = true;
    
    // when
    auto requests = llmManager->impl_->getRequests_();

    // then
    ASSERT_EQ(requests.size(), 1);
    ASSERT_EQ(requests[0]->requestId, "req_1");
    ASSERT_TRUE(requests[0]->ignoreEos.has_value());
    ASSERT_EQ(requests[0]->ignoreEos.value(), ignoreEos);
}

TEST_F(LlmManagerTest, should_return_v2_stop_strings_ok_when_input_v1_stop_strings_tensor)
{
    // given
    std::map<std::string, std::string> ipInfo = {{"infer_mode", "dmi"}};
    std::shared_ptr<LlmManager> llmManager = std::make_shared<LlmManager>("configPath_", GetRequestsStubStopStrings, nullptr, nullptr, nullptr, nullptr, ipInfo);

    std::string stopStrings = "test_stopStrings_123";
    
    // when
    auto requests = llmManager->impl_->getRequests_();

    // then
    ASSERT_EQ(requests.size(), 1);
    ASSERT_EQ(requests[0]->requestId, "req_1");
    ASSERT_TRUE(requests[0]->stopStrings.has_value());
    ASSERT_EQ(requests[0]->stopStrings.value(), stopStrings);
}

TEST_F(LlmManagerTest, should_return_v2_log_probs_ok_when_input_v1_log_probs_tensor)
{
    // given
    std::map<std::string, std::string> ipInfo = {{"infer_mode", "dmi"}};
    std::shared_ptr<LlmManager> llmManager = std::make_shared<LlmManager>("configPath_", GetRequestsStubLogProbs, nullptr, nullptr, nullptr, nullptr, ipInfo);

    bool logProbs = true;
    
    // when
    auto requests = llmManager->impl_->getRequests_();

    // then
    ASSERT_EQ(requests.size(), 1);
    ASSERT_EQ(requests[0]->requestId, "req_1");
    ASSERT_TRUE(requests[0]->logprobs.has_value());
    ASSERT_EQ(requests[0]->logprobs.value(), logProbs);
}

TEST_F(LlmManagerTest, should_return_v2_top_log_probs_ok_when_input_v1_top_log_probs_tensor)
{
    // given
    std::map<std::string, std::string> ipInfo = {{"infer_mode", "dmi"}};
    std::shared_ptr<LlmManager> llmManager = std::make_shared<LlmManager>("configPath_", GetRequestsStubTopLogProbs, nullptr, nullptr, nullptr, nullptr, ipInfo);

    uint32_t topLogProbs = 123;
    
    // when
    auto requests = llmManager->impl_->getRequests_();

    // then
    ASSERT_EQ(requests.size(), 1);
    ASSERT_EQ(requests[0]->requestId, "req_1");
    ASSERT_TRUE(requests[0]->topLogprobs.has_value());
    ASSERT_EQ(requests[0]->topLogprobs.value(), topLogProbs);
}

TEST_F(LlmManagerTest, should_return_v2_temperature_ok_when_input_v1_temperature_tensor)
{
    // given
    std::map<std::string, std::string> ipInfo = {{"infer_mode", "dmi"}};
    std::shared_ptr<LlmManager> llmManager = std::make_shared<LlmManager>("configPath_", GetRequestsStubTemperature, nullptr, nullptr, nullptr, nullptr, ipInfo);

    float temperature = 123.123f;
    
    // when
    auto requests = llmManager->impl_->getRequests_();

    // then
    ASSERT_EQ(requests.size(), 1);
    ASSERT_EQ(requests[0]->requestId, "req_1");
    ASSERT_TRUE(requests[0]->temperature.has_value());
    ASSERT_EQ(requests[0]->temperature.value(), temperature);
}

TEST_F(LlmManagerTest, should_return_v2_top_k_ok_when_input_v1_top_k_tensor)
{
    // given
    std::map<std::string, std::string> ipInfo = {{"infer_mode", "dmi"}};
    std::shared_ptr<LlmManager> llmManager = std::make_shared<LlmManager>("configPath_", GetRequestsStubTopK, nullptr, nullptr, nullptr, nullptr, ipInfo);

    int32_t topK = 12;
    
    // when
    auto requests = llmManager->impl_->getRequests_();

    // then
    ASSERT_EQ(requests.size(), 1);
    ASSERT_EQ(requests[0]->requestId, "req_1");
    ASSERT_TRUE(requests[0]->topK.has_value());
    ASSERT_EQ(requests[0]->topK.value(), topK);
}

TEST_F(LlmManagerTest, should_return_v2_top_p_ok_when_input_v1_top_p_tensor)
{
    // given
    std::map<std::string, std::string> ipInfo = {{"infer_mode", "dmi"}};
    std::shared_ptr<LlmManager> llmManager = std::make_shared<LlmManager>("configPath_", GetRequestsStubTopP, nullptr, nullptr, nullptr, nullptr, ipInfo);

    float topP = 12.12f;
    
    // when
    auto requests = llmManager->impl_->getRequests_();

    // then
    ASSERT_EQ(requests.size(), 1);
    ASSERT_EQ(requests[0]->requestId, "req_1");
    ASSERT_TRUE(requests[0]->topP.has_value());
    ASSERT_EQ(requests[0]->topP.value(), topP);
}

TEST_F(LlmManagerTest, should_return_v2_typical_p_ok_when_input_v1_typical_p_tensor)
{
    // given
    std::map<std::string, std::string> ipInfo = {{"infer_mode", "dmi"}};
    std::shared_ptr<LlmManager> llmManager = std::make_shared<LlmManager>("configPath_", GetRequestsStubTypicalP, nullptr, nullptr, nullptr, nullptr, ipInfo);

    float typicalP = 1.1f;
    
    // when
    auto requests = llmManager->impl_->getRequests_();

    // then
    ASSERT_EQ(requests.size(), 1);
    ASSERT_EQ(requests[0]->requestId, "req_1");
    ASSERT_EQ(requests[0]->typicalP, typicalP);
}

TEST_F(LlmManagerTest, should_return_v2_do_sample_ok_when_input_v1_do_sample_tensor)
{
    // given
    std::map<std::string, std::string> ipInfo = {{"infer_mode", "dmi"}};
    std::shared_ptr<LlmManager> llmManager = std::make_shared<LlmManager>("configPath_", GetRequestsStubDoSample, nullptr, nullptr, nullptr, nullptr, ipInfo);

    bool doSample = true;
    
    // when
    auto requests = llmManager->impl_->getRequests_();

    // then
    ASSERT_EQ(requests.size(), 1);
    ASSERT_EQ(requests[0]->requestId, "req_1");
    ASSERT_TRUE(requests[0]->doSample.has_value());
    ASSERT_EQ(requests[0]->doSample.value(), doSample);
}

TEST_F(LlmManagerTest, should_return_v2_seed_ok_when_input_v1_seed_tensor)
{
    // given
    std::map<std::string, std::string> ipInfo = {{"infer_mode", "dmi"}};
    std::shared_ptr<LlmManager> llmManager = std::make_shared<LlmManager>("configPath_", GetRequestsStubSeed, nullptr, nullptr, nullptr, nullptr, ipInfo);

    uint64_t seed = 123;
    
    // when
    auto requests = llmManager->impl_->getRequests_();

    // then
    ASSERT_EQ(requests.size(), 1);
    ASSERT_EQ(requests[0]->requestId, "req_1");
    ASSERT_EQ(requests[0]->seed, seed);
}

TEST_F(LlmManagerTest, should_return_v2_repetition_penalty_ok_when_input_v1_repetition_penalty_tensor)
{
    // given
    std::map<std::string, std::string> ipInfo = {{"infer_mode", "dmi"}};
    std::shared_ptr<LlmManager> llmManager = std::make_shared<LlmManager>("configPath_", GetRequestsStubRepetitionPenalty, nullptr, nullptr, nullptr, nullptr, ipInfo);

    float repetitionPenalty = 1.2f;
    
    // when
    auto requests = llmManager->impl_->getRequests_();

    // then
    ASSERT_EQ(requests.size(), 1);
    ASSERT_EQ(requests[0]->requestId, "req_1");
    ASSERT_TRUE(requests[0]->repetitionPenalty.has_value());
    ASSERT_EQ(requests[0]->repetitionPenalty.value(), repetitionPenalty);
}

TEST_F(LlmManagerTest, should_return_v2_frequency_penalty_ok_when_input_v1_frequency_penalty_tensor)
{
    // given
    std::map<std::string, std::string> ipInfo = {{"infer_mode", "dmi"}};
    std::shared_ptr<LlmManager> llmManager = std::make_shared<LlmManager>("configPath_", GetRequestsStubFrequencyPenalty, nullptr, nullptr, nullptr, nullptr, ipInfo);

    float frequencyPenalty = 1.3f;
    
    // when
    auto requests = llmManager->impl_->getRequests_();

    // then
    ASSERT_EQ(requests.size(), 1);
    ASSERT_EQ(requests[0]->requestId, "req_1");
    ASSERT_TRUE(requests[0]->frequencyPenalty.has_value());
    ASSERT_EQ(requests[0]->frequencyPenalty.value(), frequencyPenalty);
}

TEST_F(LlmManagerTest, should_return_v2_presency_penalty_ok_when_input_v1_presency_penalty_tensor)
{
    // given
    std::map<std::string, std::string> ipInfo = {{"infer_mode", "dmi"}};
    std::shared_ptr<LlmManager> llmManager = std::make_shared<LlmManager>("configPath_", GetRequestsStubPresencyPenalty, nullptr, nullptr, nullptr, nullptr, ipInfo);

    float presencyPenalty = 1.4f;
    
    // when
    auto requests = llmManager->impl_->getRequests_();

    // then
    ASSERT_EQ(requests.size(), 1);
    ASSERT_EQ(requests[0]->requestId, "req_1");
    ASSERT_TRUE(requests[0]->presencyPenalty.has_value());
    ASSERT_EQ(requests[0]->presencyPenalty.value(), presencyPenalty);
}

TEST_F(LlmManagerTest, should_return_v2_include_stop_str_ok_when_input_v1_include_stop_str_tensor)
{
    // given
    std::map<std::string, std::string> ipInfo = {{"infer_mode", "dmi"}};
    std::shared_ptr<LlmManager> llmManager = std::make_shared<LlmManager>("configPath_", GetRequestsStubIncludeStopStrInOutput, nullptr, nullptr, nullptr, nullptr, ipInfo);

    bool includeStopStrInOutput = true;
    
    // when
    auto requests = llmManager->impl_->getRequests_();

    // then
    ASSERT_EQ(requests.size(), 1);
    ASSERT_EQ(requests[0]->requestId, "req_1");
    ASSERT_TRUE(requests[0]->includeStopStrInOutput.has_value());
    ASSERT_EQ(requests[0]->includeStopStrInOutput.value(), includeStopStrInOutput);
}

TEST_F(LlmManagerTest, should_return_v2_watermark_ok_when_input_v1_watermark_tensor)
{
    // given
    std::map<std::string, std::string> ipInfo = {{"infer_mode", "dmi"}};
    std::shared_ptr<LlmManager> llmManager = std::make_shared<LlmManager>("configPath_", GetRequestsStubWatermark, nullptr, nullptr, nullptr, nullptr, ipInfo);

    bool watermark = true;
    
    // when
    auto requests = llmManager->impl_->getRequests_();

    // then
    ASSERT_EQ(requests.size(), 1);
    ASSERT_EQ(requests[0]->requestId, "req_1");
    ASSERT_EQ(requests[0]->watermark, watermark);
}

TEST_F(LlmManagerTest, should_return_v2_n_ok_when_input_v1_n_tensor)
{
    // given
    std::map<std::string, std::string> ipInfo = {{"infer_mode", "dmi"}};
    std::shared_ptr<LlmManager> llmManager = std::make_shared<LlmManager>("configPath_", GetRequestsStubN, nullptr, nullptr, nullptr, nullptr, ipInfo);

    uint32_t n = 1234;
    
    // when
    auto requests = llmManager->impl_->getRequests_();

    // then
    ASSERT_EQ(requests.size(), 1);
    ASSERT_EQ(requests[0]->requestId, "req_1");
    ASSERT_TRUE(requests[0]->n.has_value());
    ASSERT_EQ(requests[0]->n.value(), n);
}

TEST_F(LlmManagerTest, should_return_v2_best_of_ok_when_input_v1_best_of_tensor)
{
    // given
    std::map<std::string, std::string> ipInfo = {{"infer_mode", "dmi"}};
    std::shared_ptr<LlmManager> llmManager = std::make_shared<LlmManager>("configPath_", GetRequestsStubBestOf, nullptr, nullptr, nullptr, nullptr, ipInfo);

    uint32_t bestOf = 1234;
    
    // when
    auto requests = llmManager->impl_->getRequests_();

    // then
    ASSERT_EQ(requests.size(), 1);
    ASSERT_EQ(requests[0]->requestId, "req_1");
    ASSERT_TRUE(requests[0]->bestOf.has_value());
    ASSERT_EQ(requests[0]->bestOf.value(), bestOf);
}

TEST_F(LlmManagerTest, should_return_v2_use_beam_search_ok_when_input_v1_use_beam_search_tensor)
{
    // given
    std::map<std::string, std::string> ipInfo = {{"infer_mode", "dmi"}};
    std::shared_ptr<LlmManager> llmManager = std::make_shared<LlmManager>("configPath_", GetRequestsStubUseBeamSearch, nullptr, nullptr, nullptr, nullptr, ipInfo);

    bool useBeamSearch = true;
    
    // when
    auto requests = llmManager->impl_->getRequests_();

    // then
    ASSERT_EQ(requests.size(), 1);
    ASSERT_EQ(requests[0]->requestId, "req_1");
    ASSERT_TRUE(requests[0]->useBeamSearch.has_value());
    ASSERT_EQ(requests[0]->useBeamSearch.value(), useBeamSearch);
}

TEST_F(LlmManagerTest, should_return_v1_output_ids_ok_when_input_v2_response_ids_tensor)
{
    // given
    std::map<std::string, std::string> ipInfo = {{"infer_mode", "dmi"}};
    std::shared_ptr<LlmManager> llmManager = std::make_shared<LlmManager>("configPath_", nullptr, SetResponseSpy, nullptr, nullptr, nullptr, ipInfo);
    std::shared_ptr<Response> responsev2 = std::make_shared<Response>("req_1");
    ResponseContent content;
    content.seqId = 1;
    content.parentSeqId = 1;
    content.outTokenIds.push_back(4);
    content.outTokenIds.push_back(5);
    content.outTokenIds.push_back(6);
    content.outLogProbs.push_back(1.0f);
    content.outLogProbs.push_back(2.0f);
    content.outLogProbs.push_back(3.0f);
    content.topLogProbTokenIds.push_back(1);
    content.topLogProbTokenIds.push_back(2);
    content.topLogProbTokenIds.push_back(3);
    content.topLogProbs.push_back(4.0f);
    content.topLogProbs.push_back(5.0f);
    content.topLogProbs.push_back(6.0f);
    content.srcBlockTable.push_back(std::vector<int64_t>{6});
    responsev2->metrics.batchSize = 6;
    responsev2->metrics.queueWaitTime = 8;
    responsev2->responseContents.push_back(content);
    responsev2->numParallelTokens = 1;
    // when
    llmManager->impl_->handleResponse_({responsev2});
    // then
    ASSERT_EQ(GetResponseSpy("req_1").has_value(), true);
    auto responseInfo = GetResponseSpy("req_1").value();
    ASSERT_EQ(responseInfo.isFinal, false);
    auto tensors = responseInfo.outputs;
    ASSERT_EQ(tensors.size(), 10);
    auto tensor = tensors["OUTPUT_IDS"];
    ASSERT_EQ(tensor->dataType, InferDataType::TYPE_INT64);
    ASSERT_EQ(tensor->dataShape, std::vector<int64_t>({1, 3}));
    std::vector<int64_t> expectedOutputIds = {4, 5, 6};
    ASSERT_EQ(std::vector<int64_t>(static_cast<int64_t *>(tensor->data), static_cast<int64_t *>(tensor->data) + tensor->dataShape[1]), expectedOutputIds);
}

} // namespace mindie_llm