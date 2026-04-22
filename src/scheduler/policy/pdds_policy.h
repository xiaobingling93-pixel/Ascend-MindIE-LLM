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

#ifndef PD_SEPERATE_POLICY_H
#define PD_SEPERATE_POLICY_H

#include "fcfs_policy.h"
#include "scheduler.h"
namespace mindie_llm {
/**
 * PDDS = prefill-decode decoupled scheduling
 * P Node: request --> waiting queue -->running queue --> transfer queue, in future KV may swap out to CPU
 * D Node: request --> waiting queue --> transfer queue --> running queue,
           If the sequence group of the KV is pulled, the sequence group will be moved
           from the transfer queue to the running queue.
 * As Scheduler and KV transmission are independent,
 */
class PDDSPolicy final : public FcfsPolicy {
   public:
    PDDSPolicy(std::shared_ptr<SchedulerConfig> &schedulerConfig, BlockSpaceManagerSPtr &blockManager, Role role);

    PolicyOutput Apply(SchedulingBudget &budget, std::shared_ptr<SeqGroupCollection> &collection) override;

   private:
    Role role_;
    PolicyOutput ScheduleChunkedPrefill(SchedulingBudget &budget) override;
};

class KVTransferSchedulePolicy : public KVTransferPolicy {
   public:
    KVTransferSchedulePolicy(std::shared_ptr<SchedulerConfig> &schedulerConfig, BlockSpaceManagerSPtr &blockManager,
                             Role role);

    KVTransferPolicyOutput Apply(SchedulingBudget &budget, std::shared_ptr<SeqGroupCollection> &collection) override;

   protected:
    KVTransferPolicyOutput ScheduleKVTransferInD(SchedulingBudget &budget, bool enableChunking = false);

    std::vector<SequenceGroupSPtr> PickPublishSeqGroup();

    std::vector<ScheduledSequenceGroupSPtr> PickPullSeqGroup(SchedulingBudget &budget, bool enableChunking = false);

    std::vector<SequenceGroupSPtr> PickPulledSeqGroup();

    KVTransferPolicyOutput BuildTransferPolicyOutput(std::vector<ScheduledSequenceGroupSPtr> &pullSeqGroups) const;

   private:
    PolicyHelper policyHelper_;

    std::shared_ptr<SchedulerConfig> schedulerConfig_;

    BlockSpaceManagerSPtr blockManager_;

    Role role_;

    ISeqGroupCollectionSPtr queuesCollection_;
};
}  // namespace mindie_llm

#endif
