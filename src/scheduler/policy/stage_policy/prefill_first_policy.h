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

#ifndef PREFILL_FIRST_POLICY_H
#define PREFILL_FIRST_POLICY_H

#include "concurrent_deque.h"
#include "policy/seq_group_collection.h"
#include "policy/stage_policy/stage_policy.h"

namespace mindie_llm {
class PrefillFirstPolicy final : public StagePolicy {
   public:
    explicit PrefillFirstPolicy() {};

    PDPriorityType Apply([[maybe_unused]] ConcurrentDeque<SequenceGroupSPtr> &waiting,
                         [[maybe_unused]] ConcurrentDeque<SequenceGroupSPtr> &running,
                         [[maybe_unused]] ConcurrentDeque<SequenceGroupSPtr> &swapped) override {
        return PDPriorityType::PREFILL_FIRST;
    };

   private:
};
}  // namespace mindie_llm

#endif
