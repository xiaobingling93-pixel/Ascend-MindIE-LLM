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

#ifndef IPOLICY_H
#define IPOLICY_H

#include "block_manager_interface.h"
#include "config_info.h"
#include "scheduling_budget.h"
#include "seq_group_collection.h"

namespace mindie_llm {
enum class PolicyType { FCFS, PRIORITY };

struct PolicyOutput {
    // 1. from waiting queue + swappedQueue
    std::deque<SequenceGroupSPtr> withdrewSeqGroups_;

    // (recompute data) push left, from running queue
    std::vector<SequenceGroupSPtr> preemptedSeqGroups_;

    // (prefill data) push right, from waiting queue;
    std::vector<std::shared_ptr<ScheduledSequenceGroup>> prefillSeqGroups_;

    // (decode data) from running queue, else from swapped queue(push right)
    std::vector<std::shared_ptr<ScheduledSequenceGroup>> decodeSeqGroups_;

    // (swapped data) 1. push right, from running queue
    std::vector<SequenceGroupSPtr> swappedOutSeqGroups_;

    // 1. from waiting queue; 2. from swapped queue
    std::vector<SequenceGroupSPtr> ignoredSeqGroups_;

    // from swapped output
    std::vector<std::pair<BlockId, BlockId>> blocksToSwapIn_;

    // from running output
    std::vector<std::pair<BlockId, BlockId>> blocksToSwapOut_;

    // from running output + swapped in output
    std::vector<std::pair<BlockId, BlockId>> blocksToCopy_;

    size_t numPrefillGroups{0};

    size_t numBatchedTokens{0};

    size_t numPreempted{0};
};

struct KVTransferPolicyOutput {
    // from waiting queue, need to have kv pulled
    std::vector<ScheduledSequenceGroupSPtr> pullSeqGroups;

    // from waiting queue, need to have kv pulled, but cannot be scheduled this round.
    std::vector<SequenceGroupSPtr> withdrewSeqGroups;
};

class Policy {
   public:
    Policy() = default;
    virtual ~Policy() = default;
    virtual PolicyOutput Apply(SchedulingBudget &budget, std::shared_ptr<SeqGroupCollection> &collection) = 0;
    bool newRequestFirst_{false};
};

class KVTransferPolicy {
   public:
    virtual ~KVTransferPolicy() = default;
    virtual KVTransferPolicyOutput Apply(SchedulingBudget &budget, std::shared_ptr<SeqGroupCollection> &collection) = 0;
};
}  // namespace mindie_llm

#endif
