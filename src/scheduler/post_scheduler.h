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

#ifndef MINDIE_LLM_PRE_SCHEDULER_H
#define MINDIE_LLM_PRE_SCHEDULER_H

#include <cstddef>

#include "sequence_group.h"
#include "sequence_group_meta_data.h"

namespace mindie_llm {
struct BatchInfo {
    int64_t maxBatchSize_;

    int64_t maxSeqLen_;

    std::vector<int64_t> batchSizeList_;

    BatchInfo() = default;

    BatchInfo(int64_t maxBatchSize, int64_t maxSeqLen) : maxBatchSize_(maxBatchSize), maxSeqLen_(maxSeqLen) {}
};

class PostScheduler {
   public:
    static void SyncBatchInfo(BatchInfo &batchInfo, size_t dpRank, bool enableDistributed);

    static void SyncSeqLenList(std::vector<std::vector<int64_t>> &tokenNumList, std::vector<int64_t> &batchSizeList,
                               size_t paddingSize, size_t dpRank, bool enableDistributed);
    static void AllGatherBatchesAcrossDPs(std::vector<std::vector<SequenceGroupMetaDatas>> &allDpMetas,
                                          std::vector<std::vector<SchedulerOutputs>> &allDpOutputs, size_t dpRank);

    static std::unordered_set<SequenceId> AllGatherCleanSeqIdsAcrossDPs(
        std::unordered_set<SequenceId> &curCleanSeqIdSet, size_t dpRank);

   private:
    static void SyncBatchInfoAcrossDP(BatchInfo &batchInfo, size_t dpRank);

    static void SyncBatchInfoAcrossNodes(BatchInfo &batchInfo);

    static void SyncSeqLenListAcrossDP(std::vector<std::vector<int64_t>> &tokenNumList, size_t dpRank);

    static void SyncSeqLenListAcrossNodes(std::vector<std::vector<int64_t>> &tokenNumList);

    static void AddPaddingData(std::vector<std::vector<int64_t>> &tokenNumList, size_t paddingSize);

    static void RemovePaddingData(std::vector<std::vector<int64_t>> &tokenNumList, std::vector<int64_t> batchSizeList);
};
}  // namespace mindie_llm
#endif
