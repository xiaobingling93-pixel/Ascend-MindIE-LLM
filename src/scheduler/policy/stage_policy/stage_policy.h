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

#ifndef ISTAGE_POLICY_H
#define ISTAGE_POLICY_H

#include "basic_types.h"
#include "concurrent_deque.h"
#include "policy/seq_group_collection.h"

namespace mindie_llm {
enum class StagePolicyType { PREFILL_FIRST, FIXED_COST_TIME_TPT_FIRST, LATENCY_FIRST, EDGE_CLOUD };

class StagePolicy {
   public:
    StagePolicy() = default;
    virtual ~StagePolicy() = default;
    virtual PDPriorityType Apply(ConcurrentDeque<SequenceGroupSPtr> &waiting,
                                 ConcurrentDeque<SequenceGroupSPtr> &running,
                                 ConcurrentDeque<SequenceGroupSPtr> &swapped) = 0;
    // 在推理请求完成时标记当前时间戳
    virtual void MarkInferenceEndTimeStamp() {}
    // 在推理请求开始时标记当前时间戳
    virtual void MarkInferenceStartTimeStamp(PDPriorityType) {}
    // 在配比微调场景下，设置prefill百分比中整数部分
    virtual void SetPrefillPercentage(uint32_t) {}
    // 在配比微调场景下，根据当前等待队列、运行队列和交换队列中的请求类型，和当前prefill请求执行占比，决定当前Flex节点的角色
    virtual Role GetFlexRole(ConcurrentDeque<SequenceGroupSPtr> &, ConcurrentDeque<SequenceGroupSPtr> &,
                             ConcurrentDeque<SequenceGroupSPtr> &) {
        return Role::FlexP;
    }
};

}  // namespace mindie_llm

#endif
