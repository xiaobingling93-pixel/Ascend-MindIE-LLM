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

#ifndef TIME_DIVISION_POLICY_H
#define TIME_DIVISION_POLICY_H
#include <queue>

#include "policy/seq_group_collection.h"
#include "policy/stage_policy/stage_policy.h"

namespace mindie_llm {
/* 基于时间片轮转的调度策略
1. 记录每次推理请求的开始和结束时间戳
2. 计算在一个滑动时间窗口内，prefill和decode请求的累计推理时间
3. 根据预设的prefill百分比，决定当前时间窗口内优先调度prefill还是decode请求
4. 如果当前时间窗口内没有对应类型的请求，则调度另一类型的请求
5. 如果有prefill和decode都在本地执行的flexlocal请求，则优先调度本地请求
*/
class TimeDivisionPolicy final : public StagePolicy {
   public:
    explicit TimeDivisionPolicy() = default;

    PDPriorityType Apply(ConcurrentDeque<SequenceGroupSPtr> &waiting, ConcurrentDeque<SequenceGroupSPtr> &running,
                         ConcurrentDeque<SequenceGroupSPtr> &swapped) override;
    // 在推理请求完成时标记当前时间戳
    void MarkInferenceEndTimeStamp() override;
    // 在推理请求开始时标记当前时间戳
    void MarkInferenceStartTimeStamp(PDPriorityType type) override;

    void SetPrefillPercentage(uint32_t prefillPercentage) override;

    Role GetFlexRole(ConcurrentDeque<SequenceGroupSPtr> &waiting, ConcurrentDeque<SequenceGroupSPtr> &running,
                     ConcurrentDeque<SequenceGroupSPtr> &swapped) override;

   private:
    void UpdateSlideWindow();
    uint64_t GetNowTimeStamp() const;
    uint64_t windowDuration_{2000000};  // 2 seconds
    uint64_t prefillInferenceTime_{0};
    uint64_t decodeInferenceTime_{0};
    uint32_t prefillPercentage_{50};  // 0-100
    PDPriorityType nowType_{PDPriorityType::MIX};
    std::queue<std::pair<PDPriorityType, uint64_t>> inferenceStartTimeRecords_;
    std::queue<std::pair<PDPriorityType, uint64_t>> inferenceTimeWindow_;

    template <typename Predicate>
    bool CheckRequestExistence(ConcurrentDeque<SequenceGroupSPtr> &source, Predicate &&pred) const {
        if (source.Empty()) {
            return false;
        }

        bool found = false;
        source.ForEach(
            [&](const SequenceGroupSPtr &sgPtr) {
                if (!found && pred(sgPtr)) {  // 短路逻辑优化
                    found = true;
                }
            },
            source.Size());

        return found;
    }

    bool HasPrefillReq(ConcurrentDeque<SequenceGroupSPtr> &waiting) const {
        return CheckRequestExistence(waiting, [](const auto &sg) { return !sg->isDecode_ && !sg->isFlexLocal_; });
    }

    bool HasDecodeReq(ConcurrentDeque<SequenceGroupSPtr> &waiting, ConcurrentDeque<SequenceGroupSPtr> &running,
                      ConcurrentDeque<SequenceGroupSPtr> &swapping) const {
        return CheckRequestExistence(waiting, [](const auto &sg) { return sg->isDecode_; }) ||
               CheckRequestExistence(running, [](const auto &sg) { return sg->isDecode_; }) ||
               CheckRequestExistence(swapping, [](const auto &sg) { return sg->isDecode_; });
    }

    bool HasFlexLocalReq(ConcurrentDeque<SequenceGroupSPtr> &waiting, ConcurrentDeque<SequenceGroupSPtr> &running,
                         ConcurrentDeque<SequenceGroupSPtr> &swapping) const {
        return CheckRequestExistence(waiting, [](const auto &sg) { return sg->isFlexLocal_; }) ||
               CheckRequestExistence(running, [](const auto &sg) { return sg->isFlexLocal_; }) ||
               CheckRequestExistence(swapping, [](const auto &sg) { return sg->isFlexLocal_; });
    }
};
}  // namespace mindie_llm

#endif
