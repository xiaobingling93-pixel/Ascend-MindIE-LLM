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

#include "time_division_policy.h"

#include <cmath>

namespace mindie_llm {
void TimeDivisionPolicy::MarkInferenceEndTimeStamp() {
    uint64_t endTime = GetNowTimeStamp();
    // 计算当前时间与上次记录的时间差
    if (inferenceStartTimeRecords_.empty()) {
        // 理论上执行推理完成时应该有开始的记录
        return;
    } else {
        uint64_t lastStartTime = inferenceStartTimeRecords_.front().second;
        uint64_t inferenceTime = endTime - lastStartTime;
        auto type = inferenceStartTimeRecords_.front().first;
        if (type == PDPriorityType::PREFILL_FIRST) {
            prefillInferenceTime_ += inferenceTime;
        }
        if (type == PDPriorityType::DECODE_FIRST) {
            decodeInferenceTime_ += inferenceTime;
        }
        inferenceStartTimeRecords_.pop();
        inferenceTimeWindow_.push(std::make_pair(type, inferenceTime));
    }
    UpdateSlideWindow();
}

void TimeDivisionPolicy::MarkInferenceStartTimeStamp(PDPriorityType type) {
    inferenceStartTimeRecords_.push(std::make_pair(type, GetNowTimeStamp()));
}

void TimeDivisionPolicy::UpdateSlideWindow() {
    auto totalTime = prefillInferenceTime_ + decodeInferenceTime_;
    while (inferenceTimeWindow_.size() != 0 && totalTime > windowDuration_) {
        auto [type, inferTime] = inferenceTimeWindow_.front();
        if (type == PDPriorityType::PREFILL_FIRST && prefillInferenceTime_ >= inferTime) {
            prefillInferenceTime_ -= inferTime;
        } else if (decodeInferenceTime_ >= inferTime) {
            decodeInferenceTime_ -= inferTime;
        }
        if (totalTime > inferTime) {
            totalTime -= inferTime;
        }
        inferenceTimeWindow_.pop();
    }
}

PDPriorityType TimeDivisionPolicy::Apply(ConcurrentDeque<SequenceGroupSPtr> &waiting,
                                         ConcurrentDeque<SequenceGroupSPtr> &running,
                                         ConcurrentDeque<SequenceGroupSPtr> &swapped) {
    (void)running;
    (void)swapped;
    uint64_t totalTime = prefillInferenceTime_ + decodeInferenceTime_;
    constexpr uint32_t PERCENTAGE_NUM = 100;
    PDPriorityType type = PDPriorityType::DECODE_FIRST;
    if (prefillInferenceTime_ * PERCENTAGE_NUM > totalTime * prefillPercentage_) {
        type = PDPriorityType::DECODE_FIRST;
    } else {
        type = PDPriorityType::PREFILL_FIRST;
    }

    if (!HasPrefillReq(waiting) && HasDecodeReq(waiting, running, swapped) && type == PDPriorityType::PREFILL_FIRST) {
        type = PDPriorityType::DECODE_FIRST;
    } else if (HasPrefillReq(waiting) && !HasDecodeReq(waiting, running, swapped) &&
               type == PDPriorityType::DECODE_FIRST) {
        type = PDPriorityType::PREFILL_FIRST;
    }
    nowType_ = type;
    return type;
}

uint64_t TimeDivisionPolicy::GetNowTimeStamp() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

void TimeDivisionPolicy::SetPrefillPercentage(uint32_t prefillPercentage) { prefillPercentage_ = prefillPercentage; }

Role TimeDivisionPolicy::GetFlexRole(ConcurrentDeque<SequenceGroupSPtr> &waiting,
                                     ConcurrentDeque<SequenceGroupSPtr> &running,
                                     ConcurrentDeque<SequenceGroupSPtr> &swapped) {
    PDPriorityType type = Apply(waiting, running, swapped);
    Role role = Role::FlexPnD;
    switch (type) {
        case PDPriorityType::PREFILL_FIRST:
            role = Role::FlexP;
            break;

        case PDPriorityType::DECODE_FIRST:
            role = Role::FlexD;
            break;
        default:
            break;
    }

    if (HasFlexLocalReq(waiting, running, swapped) && !HasPrefillReq(waiting) &&
        !HasDecodeReq(waiting, running, swapped)) {
        role = Role::FlexPnD;
    }

    return role;
}

}  // namespace mindie_llm
