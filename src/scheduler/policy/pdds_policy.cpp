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
#include "pdds_policy.h"

#include <stdexcept>

namespace mindie_llm {
PDDSPolicy::PDDSPolicy(std::shared_ptr<SchedulerConfig> &schedulerConfig, BlockSpaceManagerSPtr &blockManager,
                       Role role)
    : FcfsPolicy(schedulerConfig, blockManager), role_(role) {
    if (schedulerConfig == nullptr) {
        throw std::invalid_argument("schedulerConfig is null");
    }
    if (blockManager == nullptr) {
        throw std::invalid_argument("blockManager is null");
    }
}

PolicyOutput PDDSPolicy::Apply(SchedulingBudget &budget, std::shared_ptr<SeqGroupCollection> &collection) {
    if (collection == nullptr) {
        throw std::invalid_argument("collection is null");
    }
    PreprocessCollection(collection);

    switch (role_) {
        case Role::P:
            if (schedulerConfig_->enableChunkedPrefill) {
                return ScheduleChunkedPrefill(budget);
            }
            // fall through to default case
            return SchedulePrefill(budget);
        case Role::D:
            return ScheduleDecode(budget);
        default:
            throw std::runtime_error("the role is not support. role = " + std::to_string(static_cast<uint8_t>(role_)));
    }
}

PolicyOutput PDDSPolicy::ScheduleChunkedPrefill(SchedulingBudget &budget) {
    PrefillOutputs prefillOut;
    RunningOutputs runningOut;
    SwappedInOutputs swappedInOut;

    budget.statistics4PartialPrefill_ = Statistics4PartialPrefill::FromQueues(
        queuesCollection_->running_, queuesCollection_->waiting_, schedulerConfig_);

    // 1. do fcfs in waiting queue and running queue
    runningOut = ApplyToRunningQueue(budget, true);
    prefillOut = ApplyToWaitingQueue(budget, true);

    // 2. build policy out
    return BuildPolicyOutput(prefillOut, runningOut, swappedInOut, budget);
}

// the inplemetation of KVTransferSchedulePolicy
KVTransferSchedulePolicy::KVTransferSchedulePolicy(std::shared_ptr<SchedulerConfig> &schedulerConfig,
                                                   BlockSpaceManagerSPtr &blockManager, Role role)
    : policyHelper_(schedulerConfig, blockManager),
      schedulerConfig_(schedulerConfig),
      blockManager_(blockManager),
      role_(role) {
    if (schedulerConfig == nullptr) {
        throw std::invalid_argument("schedulerConfig is null");
    }
    if (blockManager == nullptr) {
        throw std::invalid_argument("blockManager is null");
    }
}

KVTransferPolicyOutput KVTransferSchedulePolicy::Apply(SchedulingBudget &budget,
                                                       std::shared_ptr<SeqGroupCollection> &collection) {
    if (collection == nullptr) {
        throw std::invalid_argument("collection is null");
    }
    this->queuesCollection_ = collection;

    return ScheduleKVTransferInD(budget);
}

KVTransferPolicyOutput KVTransferSchedulePolicy::ScheduleKVTransferInD(SchedulingBudget &budget, bool enableChunking) {
    std::vector<ScheduledSequenceGroupSPtr> pullSeqGroups = PickPullSeqGroup(budget, enableChunking);

    return BuildTransferPolicyOutput(pullSeqGroups);
}

std::vector<ScheduledSequenceGroupSPtr> KVTransferSchedulePolicy::PickPullSeqGroup(SchedulingBudget &budget,
                                                                                   bool enableChunking) {
    std::vector<ScheduledSequenceGroupSPtr> pullSeqGroups;

    while (!queuesCollection_->waiting_.empty()) {
        SequenceGroupSPtr seqGroup = queuesCollection_->waiting_.front();
        const auto [numNewTokensUncached, numNewTokensCached] = policyHelper_.GetNumComputeNewUnCachedAndCachedTokens(
            seqGroup, SequenceStatus::WAITING, enableChunking, budget);
        const size_t numNewTokens = numNewTokensUncached + numNewTokensCached;

        size_t freeNpuBlocksNum = blockManager_->GetNumFreeNpuBlocks();
        size_t totalNpuBlocksNum = blockManager_->GetTotalNpuBlocks();
        size_t preservedNpuBlocks = static_cast<size_t>(PRESERVED_FACTOR_FOR_DECODE * totalNpuBlocksNum);
        if (freeNpuBlocksNum < preservedNpuBlocks) {
            break;
        }

        // 1. don't check the prompt limit, it is pointless
        // 2. check the prompt can be allocate
        const auto canAllocate = blockManager_->CanAllocate(seqGroup);
        if (canAllocate == AllocStatus::LATER) {
            break;
        } else if (canAllocate == AllocStatus::NEVER) {
            // 如果prompt长度很长，则在add request的时候就做长度校验，返回添加失败。
            throw std::runtime_error("Prompt sequence too long.");
        }

        // 3. check the tokens and seqs
        const int numNewSeqs = seqGroup->GetMaxNumRunningSeqs();
        if (!budget.CanSchedule(numNewTokensUncached, numNewSeqs)) {
            break;
        }

        // 4. pop seqgroup and allocate, then update sequence status
        queuesCollection_->waiting_.pop_front();
        policyHelper_.AllocateAndSetRunning(seqGroup);
        pullSeqGroups.emplace_back(std::make_shared<ScheduledSequenceGroup>(seqGroup, numNewTokens));

        // 5. update budget
        budget.AddNumBatchedTokens(seqGroup->requestId, numNewTokensUncached, numNewTokensCached);
        budget.AddNumSeqs(seqGroup->requestId, numNewSeqs);
    }

    return pullSeqGroups;
}

KVTransferPolicyOutput KVTransferSchedulePolicy::BuildTransferPolicyOutput(
    std::vector<ScheduledSequenceGroupSPtr> &pullSeqGroups) const {
    KVTransferPolicyOutput output;
    output.pullSeqGroups = std::move(pullSeqGroups);
    output.withdrewSeqGroups.insert(output.withdrewSeqGroups.end(), queuesCollection_->waiting_.begin(),
                                    queuesCollection_->waiting_.end());
    return output;
}
}  // namespace mindie_llm
