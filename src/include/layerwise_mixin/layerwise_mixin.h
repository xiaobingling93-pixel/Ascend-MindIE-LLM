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

#ifndef LAYERWISE_MIXIN_H
#define LAYERWISE_MIXIN_H

#include <deque>
#include <memory>
#include "basic_types.h"
#include "sequence_group.h"
#include "block_manager_interface.h"
#include "policy/stage_policy/stage_policy.h"

namespace mindie_llm {
class LayerwiseMixin {
public:
    void LwdPrepareBatch(bool layerwiseDisaggregated, SchedulerOutputs &scheduleOut) const;

    void LwdEngineAddBatchCnt(bool layerwiseDisaggregated, std::shared_ptr<StagePolicy> stagePolicy,
        SchedulerOutputs &scheduleOut) const;

    void LwdComputeArrTimeGap(bool layerwiseDisaggregated, SequenceGroupSPtr &seqGroup,
        SequenceGroupSPtr lastSeqGroup);

    void LwdSetRecomputeArrTime(bool layerwiseDisaggregated, SequenceGroupSPtr &seqGroup,
        SequenceGroupSPtr lastSeqGroup);

    bool LwdProcessResponse(bool layerwiseDisaggregated, SequenceGroupSPtr seqGroup, ForwardMode &lastForwardMode,
        ForwardMode lwdCurrBatchType, std::deque<SequenceGroupSPtr> &recomputeInDBatchQueue) const;
    void LwdProcessRecomputeSeq(bool layerwiseNeedUpdate, ForwardMode lastForwardMode,
                                   const std::deque<SequenceGroupSPtr> &recomputeInDBatchQueue) const;
    void LwdHandlerSubBatchCnt(bool layerwiseNeedUpdate, std::shared_ptr<StagePolicy> stagePolicy,
        ForwardMode lastForwardMode) const;

    void LwdWaitingResponse(PDPriorityType pdPriorityType, std::shared_ptr<StagePolicy> stagePolicy);

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> lastArriveTime_;
};
}

#endif