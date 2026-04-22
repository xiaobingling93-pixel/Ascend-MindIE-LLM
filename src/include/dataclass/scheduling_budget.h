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

#ifndef SCHEDULING_BUDGET_H
#define SCHEDULING_BUDGET_H

#include <deque>
#include <memory>
#include <unordered_set>

#include "basic_types.h"
#include "config_info.h"
#include "sequence_group.h"

namespace mindie_llm {

class Statistics4PartialPrefill {
   public:
    Statistics4PartialPrefill(int numSchedulablePrefills, int numLongPrefills,
                              std::shared_ptr<SchedulerConfig> schedulerConfig);

    bool CanSchedule(const std::shared_ptr<SequenceGroup> &seqGroup) const;

    // 当waiting队列中的seq_group是长序列是，longPrefills_会+1
    void MaybeIncrementPartialPrefills(const std::shared_ptr<SequenceGroup> &seqGroup);

    static std::unique_ptr<Statistics4PartialPrefill> FromQueues(
        const std::deque<std::shared_ptr<SequenceGroup>> &running,
        const std::deque<std::shared_ptr<SequenceGroup>> &waiting, std::shared_ptr<SchedulerConfig> schedulerConfig);

    size_t SchedulablePrefills() const;
    size_t LongPrefills() const;

   private:
    size_t numSchedulablePrefills_{0};
    size_t numLongPrefills_{};
    std::shared_ptr<SchedulerConfig> schedulerConfig_;
};

using Statistics4PartialPrefillPtr = std::unique_ptr<Statistics4PartialPrefill>;

struct SchedulingBudget {
    size_t maxNumSeqs_{};

    uint32_t maxNumBatchedTokens_{};

    size_t numCachedTokens_{};

    size_t numBatchedTokens_{};

    size_t numCurSeqs_{};

    std::unordered_set<RequestId> requestIdsNumBatchedTokens_;

    std::unordered_set<RequestId> requestIdsNumCurSeqs_;

    SchedulerConfigSPtr schedulerConfig_;

    std::vector<size_t> partialPrefillOccupiedBudgetPerSlot_;

    Statistics4PartialPrefillPtr statistics4PartialPrefill_;

    SchedulingBudget(const size_t maxNumBatchedTokens, const size_t maxNumSeqs,
                     const SchedulerConfigSPtr &schedulerConfig = nullptr);

    [[nodiscard]] bool CanSchedule(const size_t numNewTokens, const size_t numNewSeqs) const;

    size_t RemainingTokenBudget() const;

    size_t GetPrefillSlots();

    void AddNumBatchedTokens(RequestId &reqId, const size_t numBatchedTokens, const size_t numCachedTokens = 0);

    void AddNumSeqs(RequestId &reqId, const size_t numCurSeqs);

    void SubtractNumBatchedTokens(const RequestId &reqId, const size_t numBatchedTokens);

    void SubtractNumSeqs(const RequestId &reqId, const size_t numCurSeqs);
};

}  // namespace mindie_llm
#endif
