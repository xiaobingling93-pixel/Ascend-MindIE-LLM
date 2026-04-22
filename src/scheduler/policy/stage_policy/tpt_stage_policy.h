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

#ifndef TPT_STAGE_POLICY_H
#define TPT_STAGE_POLICY_H

#include <stdexcept>

#include "concurrent_deque.h"
#include "config_info.h"
#include "policy/seq_group_collection.h"
#include "policy/stage_policy/stage_policy.h"
#include "sequence_group.h"

namespace mindie_llm {
class TptStagePolicy final : public StagePolicy {
   public:
    explicit TptStagePolicy(const SchedulerConfigSPtr schedulerConfig);

    PDPriorityType Apply(ConcurrentDeque<SequenceGroupSPtr> &waiting, ConcurrentDeque<SequenceGroupSPtr> &running,
                         ConcurrentDeque<SequenceGroupSPtr> &swapped) override;

   private:
    SchedulerConfigSPtr schedulerConfig_;
    uint64_t decodeWasteTime_{0};
};
}  // namespace mindie_llm

#endif
