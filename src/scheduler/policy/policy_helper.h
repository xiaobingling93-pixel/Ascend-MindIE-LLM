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

#ifndef POLICY_HELPER_H
#define POLICY_HELPER_H

#include "block_manager_interface.h"
#include "config_info.h"
#include "scheduling_budget.h"

namespace mindie_llm {
class PolicyHelper {
   public:
    PolicyHelper(std::shared_ptr<SchedulerConfig> schedulerConfig, BlockSpaceManagerSPtr blockManager);

    std::pair<size_t, size_t> GetNumComputeNewUnCachedAndCachedTokens(const SequenceGroupSPtr seqGroup,
                                                                      const SequenceStatus status, bool enableChunking,
                                                                      SchedulingBudget &budget);

    size_t GetPromptLimit(SequenceGroupSPtr seqGroup, SchedulingBudget &budget) const;

    void AllocateAndSetRunning(SequenceGroupSPtr seqGroup) const;

    bool CanAppendSlots(SequenceGroupSPtr seqGroup) const;

    void AppendSlotForSeqs(std::vector<SequenceSPtr> parallelSeqs, RequestId reqId,
                           std::vector<std::pair<BlockId, BlockId>> &blockToCopy);

    void AppendSlots(const SequenceGroupSPtr seqGroup, std::vector<std::pair<BlockId, BlockId>> &blockToCopy);

    AllocStatus CanSwapIn(SequenceGroupSPtr seqGroup);

    void SwapIn(SequenceGroupSPtr seqGroup, std::vector<std::pair<BlockId, BlockId>> &blockToSwapIn) const;

    bool CanSwapOut(SequenceGroupSPtr seqGroup);

    void SwapOut(SequenceGroupSPtr seqGroup, std::vector<std::pair<BlockId, BlockId>> &blockToSwapOut);

    void ForkSeq(SequenceSPtr parentSeq, SequenceSPtr &childSeq) const;

    void FreeSeq(SequenceSPtr seq) const;

    void FreeSeqGroup(SequenceGroupSPtr seqGroup) const;

    size_t GetNumComputeChunkSize(SchedulingBudget &budget, size_t numNewTokens) const;

   private:
    void SeqsSwapIn(SequenceGroupSPtr seqGroup, std::vector<std::pair<BlockId, BlockId>> &blockToSwapIn) const;
    void SeqsSwapOut(SequenceGroupSPtr seqGroup, std::vector<std::pair<BlockId, BlockId>> &blockToSwapOut);

    std::shared_ptr<SchedulerConfig> schedulerConfig_;

    BlockSpaceManagerSPtr blockManager_;

    void validateSequenceGroup(const SequenceGroupSPtr &seqGroup, const std::string &context) const;
};
}  // namespace mindie_llm

#endif
