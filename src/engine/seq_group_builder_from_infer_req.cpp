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
#include <functional>
#include "log.h"
#include "request_response/request.h"
#include "request_response/request_id.h"
#include "common_util.h"
#include "seq_group_builder_from_infer_req.h"

namespace mindie_llm {
SequenceGroupSPtr SeqGroupBuilderFromInferReq::BuildSequenceGroup(RequestSPtr request,
                                                                  SchedulerConfigSPtr schedulerConfig,
                                                                  size_t rankId)
{
    // 设置sampling参数
    SamplingParamsSPtr sampleParamSptr =
        GetSampleParamFromInferRequest(request, true);
    sampleParamSptr->enableParallelSampling = (sampleParamSptr->useBeamsearch || sampleParamSptr->bestOf > 1);

    // 创建sequence，当前一个SequenceGroup只有一个Sequence
    std::vector<SequenceSPtr> seqs;
    SequenceSPtr seqSptr = InitSeqFromInferRequest(request, schedulerConfig);
    seqs.push_back(seqSptr);

    // 增加sequence id到request id的映射
    std::string requestId = request->requestId;

    // 3 创建SequenceGroup并做参数初始化
    std::optional<std::string> loraId;
    if (request->loraId != "None") {
        loraId = request->loraId;
    }
    SequenceGroupSPtr seqGrpSPtr = std::make_shared<SequenceGroup>(requestId, seqs, sampleParamSptr, loraId, rankId);
    InitSeqGrpFromInferRequest(request, requestId, seqGrpSPtr, schedulerConfig);
    if (sampleParamSptr->enableParallelSampling) {
        /* 并行采样使用，存储一次采样中生成的全部序列,包括自身 */
        seqGrpSPtr->seqId2ParallelSeqGroup_.Insert(seqGrpSPtr->firstSeq->seqId_, seqGrpSPtr);
        /* 并行采样使用，根节点的父亲也是自己 */
        seqGrpSPtr->parentSeqId_ = seqGrpSPtr->firstSeq->seqId_;
    }
    return seqGrpSPtr;
}

SamplingParamsSPtr SeqGroupBuilderFromInferReq::GetSampleParamFromInferRequest(RequestSPtr request,
                                                                               bool chooseV2BlockManager)
{
    SamplingParamsSPtr sampleParamSptr = std::make_shared<SamplingParams>();
    sampleParamSptr->maxOutputLen = request->maxOutputLen;
    sampleParamSptr->temperature = request->temperature;
    sampleParamSptr->topK = request->topK;
    sampleParamSptr->topP = request->topP;
    sampleParamSptr->typicalP = request->typicalP;
    sampleParamSptr->doSample = request->doSample;
    sampleParamSptr->seed = request->seed;
    sampleParamSptr->repetitionPenalty = request->repetitionPenalty;
    sampleParamSptr->frequencyPenalty = request->frequencyPenalty;
    sampleParamSptr->presencePenalty = request->presencyPenalty;
    sampleParamSptr->watermark = request->watermark;
    sampleParamSptr->includeStopStrInOutput = request->includeStopStrInOutput;
    if (request->stopTokenIds.has_value()) {
        sampleParamSptr->stopTokenIds = request->stopTokenIds.value();
    }
    if (request->stopStrings.has_value()) {
        sampleParamSptr->stopStrings = request->stopStrings.value();
    }

    if (chooseV2BlockManager) {
        uint32_t nValue = request->n.value_or(1);
        sampleParamSptr->n = nValue;
        // 如果 request 中未指定 bestOf，则默认使用 n 的值
        sampleParamSptr->bestOf = request->bestOf.value_or(nValue);
    }
    sampleParamSptr->useBeamsearch = request->useBeamSearch.value_or(false);
    sampleParamSptr->logprobs = request->logprobs;
    sampleParamSptr->topLogprobs = request->topLogprobs;

    return sampleParamSptr;
}

SequenceSPtr SeqGroupBuilderFromInferReq::InitSeqFromInferRequest(RequestSPtr request,
                                                                  SchedulerConfigSPtr schedulerConfig)
{
    SequenceId seqId = IDUtils::GenerateSequenceId();
    // 避免随机生成的ID与虚拟推理ID冲突，若冲突则重新生成
    while (seqId == SIMULATE_SEQUENCE_ID) {
        seqId = IDUtils::GenerateSequenceId();
    }
    // 虚推请求使用固定 seqId，便于 LLM 引擎层特殊处理
    if (request->isSimulateRequest) {
        seqId = SIMULATE_SEQUENCE_ID;
        MINDIE_LLM_LOG_DEBUG("[SimulateInference] Detected simulate inference request, requestId: "
            << request->requestId << ", assigned fixed seqId: " << seqId);
    }
    // 获取prompt的token信息
    std::vector<TokenId> inputsTokenIds = request->input_ids;
    const size_t inputTokenNum = inputsTokenIds.size();
    // 判断输入的prompt是否超长
    if (inputTokenNum >= static_cast<size_t>(schedulerConfig->maxSeqLen)) {
        throw std::invalid_argument("prompt tokens too long.");
    }

    // 创建Sequence
    SequenceSPtr seqSPtr = std::make_shared<Sequence>(seqId, schedulerConfig->cacheBlockSize, inputsTokenIds);
    return seqSPtr;
}

/**
 * 从前端(service层)请求里获取由Tensor传递而来的参数, 组到SequenceGroup里
 */
void SeqGroupBuilderFromInferReq::InitSeqGrpFromInferRequest(RequestSPtr request, RequestIdNew inferReqId,
                                                             SequenceGroupSPtr &seqGroup,
                                                             SchedulerConfigSPtr schedulerConfigSptr)
{
    seqGroup->metrics_.inferReqId_ = inferReqId;
    seqGroup->priority_ = request->priority;
    seqGroup->isSynchronous_ = request->isSynchronous;
    seqGroup->pInstanceId = request->pInstanceId.has_value() ? request->pInstanceId.value() : 0;
    seqGroup->dpInstanceId_ = request->dpInstanceIds.empty() ? 0 : request->dpInstanceIds[0];
    seqGroup->pBlockTable = request->srcBlockTable;
    seqGroup->isThinking_ = request->isThinking;
    seqGroup->skipSpecialTokens_ = request->skipSpecialTokens;
    seqGroup->ignoreEos_ = request->ignoreEos;
    seqGroup->loraId_ = request->loraId;
    seqGroup->enableThinking_ = request->enableThinking.has_value() ? request->enableThinking.value() : false;
    seqGroup->thinkingBudget_ = request->thinkingBudget.has_value() ? request->thinkingBudget.value() : 0;
    seqGroup->topLogProbs_ = request->topLogprobs.has_value() ? request->topLogprobs.value() : 0;
    request->isThinking = false;
    // lora需要将lora_id的hash 告诉block manager
    std::hash<std::string> hasher;
    if (seqGroup->loraId_.has_value()) {
        HashValue hv = hasher(seqGroup->loraId_.value());
        hv |= 1; // 要确保非0
        if (seqGroup->loraId_.value() == "None") {
            seqGroup->firstSeq->SetExtraHash(INVALID_HASH_VALUE);
        } else {
            seqGroup->firstSeq->SetExtraHash(hv);
        }
    }

    uint32_t inputTokenNum = static_cast<uint32_t>(seqGroup->seqs_.at(0)->data_.promptTokenIds.size());
    seqGroup->maxIterTimes_ =
        std::min(schedulerConfigSptr->maxSeqLen - inputTokenNum, schedulerConfigSptr->maxIterTimes);
    seqGroup->maxOutputLen_ = std::min(seqGroup->maxIterTimes_, seqGroup->sampling->maxOutputLen); // 作为配置基础值
    seqGroup->sampling->maxOutputLen = seqGroup->maxOutputLen_; // 作为每次重计算后，后续还需要推的最大长度
}

} // namespace mindie_llm