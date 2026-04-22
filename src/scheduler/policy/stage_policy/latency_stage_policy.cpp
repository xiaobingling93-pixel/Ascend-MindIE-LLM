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

#include "latency_stage_policy.h"

#include <cmath>

#include "dataclass/metric.h"
#include "policy/seq_group_collection.h"

namespace mindie_llm {
LatencyStagePolicy::LatencyStagePolicy(const SchedulerConfigSPtr schedulerConfig,
                                       std::shared_ptr<LatencyPredictor> predictor,
                                       std::shared_ptr<BlockSpaceManager> blockManager)
    : schedulerConfig_(schedulerConfig), predictor_(predictor), blockManager_(blockManager) {
    counter_ = std::make_unique<QueueCounter>(schedulerConfig_, blockManager_);
    stageDeadlines_[ForwardMode::PREFILL] = schedulerConfig_->prefillExpectedTime;
    stageDeadlines_[ForwardMode::DECODE] = schedulerConfig_->decodeExpectedTime;
}

PDPriorityType LatencyStagePolicy::Apply(ConcurrentDeque<SequenceGroupSPtr> &waiting,
                                         ConcurrentDeque<SequenceGroupSPtr> &running,
                                         ConcurrentDeque<SequenceGroupSPtr> &swapped) {
    if (running.Empty()) {
        return PDPriorityType::PREFILL_FIRST;
    } else if (waiting.Empty()) {
        return PDPriorityType::DECODE_FIRST;
    };
    if (schedulerConfig_->maxPrefillBatchSize == 0) {
        MINDIE_LLM_LOG_DEBUG("[LatencyStagePolicy] maxPrefillBatchSize is zero, Selected priority: DECODE_FIRST");
        return PDPriorityType::DECODE_FIRST;
    }
    UpdateCounter(waiting, running, swapped);  // 更新Latency policy需要的参数
    auto prefillDeadline = static_cast<float>(stageDeadlines_[ForwardMode::PREFILL]);
    auto prefillProcCostTime = GetExpectProcessTime(ForwardMode::PREFILL);
    auto prefillProcWaitTime = prefillCounter_->firstSeqWaitTime;
    auto prefillLaxity = CalStageLaxity(prefillDeadline, prefillProcCostTime, prefillProcWaitTime);
    auto decodeDeadline = static_cast<float>(stageDeadlines_[ForwardMode::DECODE]);
    auto decodeProcCostTime = GetExpectProcessTime(ForwardMode::DECODE);
    auto decodeLaxity = CalStageLaxity(decodeDeadline, decodeProcCostTime, 0);

    auto priority = (prefillLaxity < decodeLaxity) ? PDPriorityType::PREFILL_FIRST : PDPriorityType::DECODE_FIRST;

    MINDIE_LLM_LOG_DEBUG("prefillDeadline: " << prefillDeadline << ", prefillProcCostTime: " << prefillProcCostTime
                                             << ", prefillProcWaitTime: " << prefillProcWaitTime);
    MINDIE_LLM_LOG_DEBUG("decodeDeadline: " << decodeDeadline << ", decodeProcCostTime: " << decodeProcCostTime);
    MINDIE_LLM_LOG_DEBUG("prefillLaxity: " << prefillLaxity << ", decodeLaxity: " << decodeLaxity);
    std::string res = (priority == PDPriorityType::PREFILL_FIRST  ? "PREFILL_FIRST"
                       : priority == PDPriorityType::DECODE_FIRST ? "DECODE_FIRST"
                       : priority == PDPriorityType::MIX          ? "MIX"
                                                                  : "UNKNOWN");
    MINDIE_LLM_LOG_DEBUG("[LatencyStagePolicy] LatencyFirst Selected priority: " << res);
    return priority;
}

float LatencyStagePolicy::GetExpectProcessTime(ForwardMode stage) {
    float costTime = 0;
    if (stage == ForwardMode::DECODE) {
        // the number of tokens to compute equals to batch size when decode stage
        BatchStats batchStats{ForwardMode::DECODE, static_cast<uint32_t>(decodeCounter_->waitBatchesCount),
                              static_cast<uint32_t>(decodeCounter_->waitBlockNum)};
        costTime = predictor_->PredictBatchExecTime(batchStats);
        MINDIE_LLM_LOG_DEBUG("decode tokens: " << batchStats.numBatchedTokens << ", block num: "
                                               << batchStats.numBatchedTokens << ", predict time: " << costTime);
    } else if (stage == ForwardMode::PREFILL) {
        BatchStats batchStats{ForwardMode::PREFILL, static_cast<uint32_t>(prefillCounter_->waitTokensCount)};
        costTime = predictor_->PredictBatchExecTime(batchStats);
        MINDIE_LLM_LOG_DEBUG("prefill tokens: " << batchStats.numBatchedTokens << ", predict time: " << costTime);
    }
    return costTime;
}

float LatencyStagePolicy::CalStageLaxity(float deadline, float processCostTime, float stageWaitTime) const {
    if (std::fabs(deadline) < 1e-6f) {
        MINDIE_LLM_LOG_DEBUG("deadline is zero");
        return 0;
    }
    float laxity = (deadline - stageWaitTime - processCostTime) / deadline;
    return laxity;
}

void LatencyStagePolicy::UpdatePrefillCounter(ConcurrentDeque<SequenceGroupSPtr> &waiting) {
    prefillCounter_ = counter_->Count(waiting, SequenceStatus::WAITING);
}

void LatencyStagePolicy::UpdateDecodeCounter(ConcurrentDeque<SequenceGroupSPtr> &running,
                                             ConcurrentDeque<SequenceGroupSPtr> &swapped) {
    auto runningResult = counter_->Count(running, SequenceStatus::RUNNING);
    auto swappedResult = counter_->Count(swapped, SequenceStatus::SWAPPED);

    decodeCounter_->Init();
    decodeCounter_->availableSeqCount = runningResult->availableSeqCount + swappedResult->availableSeqCount;
    decodeCounter_->waitTokensCount = runningResult->waitTokensCount + swappedResult->waitTokensCount;
    decodeCounter_->waitBatchesCount = runningResult->waitBatchesCount + swappedResult->waitBatchesCount;
    decodeCounter_->waitBlockNum = runningResult->waitBlockNum + swappedResult->waitBlockNum;
    decodeCounter_->totalWaitTime = runningResult->totalWaitTime + swappedResult->totalWaitTime;
    decodeCounter_->requestNum = runningResult->requestNum + swappedResult->requestNum;
    decodeCounter_->firstSeqWaitTime = std::max(runningResult->firstSeqWaitTime, swappedResult->firstSeqWaitTime);
}

void LatencyStagePolicy::UpdateCounter(ConcurrentDeque<SequenceGroupSPtr> &waiting,
                                       ConcurrentDeque<SequenceGroupSPtr> &running,
                                       ConcurrentDeque<SequenceGroupSPtr> &swapped) {
    // 更新prefill计数器（仅waiting队列）
    UpdatePrefillCounter(waiting);

    // 更新decode计数器（running + swapped队列）
    UpdateDecodeCounter(running, swapped);

    MINDIE_LLM_LOG_DEBUG("prefillCounter_:" << *prefillCounter_ << "decodeCounter_:" << *decodeCounter_);
}
}  // namespace mindie_llm
