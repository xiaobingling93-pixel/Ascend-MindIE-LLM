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

#ifndef LATENCY_STAGE_POLICY_H
#define LATENCY_STAGE_POLICY_H

#include "concurrent_deque.h"
#include "dataclass/metric.h"
#include "latency_predictor/latency_predictor.h"
#include "latency_predictor/queue_counter.h"
#include "policy/seq_group_collection.h"
#include "policy/stage_policy/stage_policy.h"
#include "scheduling_budget.h"
#include "sequence_group.h"

namespace mindie_llm {

class LatencyStagePolicy final : public StagePolicy {
   public:
    explicit LatencyStagePolicy(const SchedulerConfigSPtr schedulerConfig, std::shared_ptr<LatencyPredictor> predictor,
                                std::shared_ptr<BlockSpaceManager> blockManager);

    PDPriorityType Apply(ConcurrentDeque<SequenceGroupSPtr> &waiting, ConcurrentDeque<SequenceGroupSPtr> &running,
                         ConcurrentDeque<SequenceGroupSPtr> &swapped) override;

   private:
    SchedulerConfigSPtr schedulerConfig_;
    std::shared_ptr<LatencyPredictor> predictor_;
    BlockSpaceManagerSPtr blockManager_;  // kv cache manager
    std::unique_ptr<QueueCounter> counter_;
    uint64_t decodeWasteTime_{0};
    std::shared_ptr<SeqCounter> prefillCounter_ = std::make_shared<SeqCounter>();
    std::shared_ptr<SeqCounter> decodeCounter_ = std::make_shared<SeqCounter>();
    uint64_t alreadyWasteTime_{0};
    uint64_t maxPrefillCostTime_ = 20000;

    std::unordered_map<ForwardMode, uint64_t> stageDeadlines_;

    float CalStageLaxity(float deadline, float processCostTime, float stageWaitTime) const;

    float GetExpectProcessTime(ForwardMode stage);

    PDPriorityType SelectPolicy(ConcurrentDeque<SequenceGroupSPtr> &waiting,
                                ConcurrentDeque<SequenceGroupSPtr> &running,
                                ConcurrentDeque<SequenceGroupSPtr> &swapped);

    void UpdatePrefillCounter(ConcurrentDeque<SequenceGroupSPtr> &waiting);

    void UpdateDecodeCounter(ConcurrentDeque<SequenceGroupSPtr> &running, ConcurrentDeque<SequenceGroupSPtr> &swapped);

    void UpdateCounter(ConcurrentDeque<SequenceGroupSPtr> &waiting, ConcurrentDeque<SequenceGroupSPtr> &running,
                       ConcurrentDeque<SequenceGroupSPtr> &swapped);
};
}  // namespace mindie_llm

#endif
