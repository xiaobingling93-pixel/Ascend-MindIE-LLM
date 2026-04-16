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

#include "tpt_stage_policy.h"

namespace mindie_llm {
TptStagePolicy::TptStagePolicy(const SchedulerConfigSPtr schedulerConfig) : schedulerConfig_(schedulerConfig) {}

PDPriorityType TptStagePolicy::Apply(ConcurrentDeque<SequenceGroupSPtr> &waiting,
                                     ConcurrentDeque<SequenceGroupSPtr> &running,
                                     [[maybe_unused]] ConcurrentDeque<SequenceGroupSPtr> &swapped) {
    if (waiting.Size() > 0) {
        uint64_t prefillCostTime = schedulerConfig_->prefillTimeMsPerReq * running.Size();
        if (running.Size() <= schedulerConfig_->maxBatchSize) {
            decodeWasteTime_ +=
                schedulerConfig_->decodeTimeMsPerReq * (schedulerConfig_->maxBatchSize - running.Size());
        }
        if (prefillCostTime > decodeWasteTime_) {
            return PDPriorityType::DECODE_FIRST;
        } else {
            decodeWasteTime_ = 0;
            return PDPriorityType::PREFILL_FIRST;
        }
    } else {
        decodeWasteTime_ = 0;
        return PDPriorityType::DECODE_FIRST;
    }
    return PDPriorityType::PREFILL_FIRST;
}
}  // namespace mindie_llm
