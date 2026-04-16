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

#ifndef LOCKED_DEQUE_COUNTER_H
#define LOCKED_DEQUE_COUNTER_H

#include <iostream>

#include "block_manager_interface.h"
#include "concurrent_deque.h"
#include "config_info.h"
#include "sequence.h"
#include "sequence_group.h"

using MilliSeconds = std::chrono::milliseconds;

namespace mindie_llm {
// Available seq counter, update by ApplyPolicy
struct SeqCounter {
    size_t availableSeqCount = 0;
    size_t waitTokensCount = 0;
    size_t waitBatchesCount = 0;
    size_t waitBlockNum = 0;
    uint64_t totalWaitTime = 0;
    uint64_t requestNum = 0;
    uint64_t firstSeqWaitTime = 0;

    SeqCounter() = default;

    void Init() {
        this->availableSeqCount = 0;
        this->waitTokensCount = 0;
        this->waitBatchesCount = 0;
        this->waitBlockNum = 0;
        this->totalWaitTime = 0;
        this->requestNum = 0;
        this->firstSeqWaitTime = 0;
    }

    friend std::ostream &operator<<(std::ostream &os, const SeqCounter &counter) {
        os << "counter : availableSeqCount=" << counter.availableSeqCount
           << ", waitTokensCount=" << counter.waitTokensCount << ", waitBatchesCount=" << counter.waitBatchesCount
           << ", waitBlockNum=" << counter.waitBlockNum << ", totalWaitTime=" << counter.totalWaitTime
           << ", requestNum=" << counter.requestNum << ", firstSeqWaitTime=" << counter.firstSeqWaitTime;
        return os;
    }
};

class QueueCounter {
   public:
    QueueCounter() = delete;
    QueueCounter(const std::shared_ptr<SchedulerConfig> &schedulerConfig,
                 const std::shared_ptr<BlockSpaceManager> &blockManager);

    // 统一统计接口
    template <typename Queue>
    std::shared_ptr<SeqCounter> Count(Queue &queue, SequenceStatus status) const {
        std::shared_ptr<SeqCounter> localResult = std::make_shared<SeqCounter>();
        localResult->Init();
        auto now = std::chrono::high_resolution_clock::now();

        // 统计队列大小和首个序列等待时间
        localResult->requestNum = queue.Size();
        if (localResult->requestNum > 0 && queue.Front()) {
            // prefill采用arriveTime，decode/swapped采用lastCompletionTime
            auto arriveTime =
                queue.Front()->IsPrefill() ? queue.Front()->arriveTime : queue.Front()->lastCompletionTime;
            localResult->firstSeqWaitTime =
                static_cast<uint64_t>(std::chrono::duration_cast<MilliSeconds>(now - arriveTime).count());
        }
        queue.ForEach(
            [this, &localResult, status](SequenceGroupSPtr seqgrp) {
                if (!seqgrp) {
                    return;
                }
                auto now = std::chrono::high_resolution_clock::now();

                // 计算等待时间
                // prefill采用arriveTime，decode/swapped采用lastCompletionTime
                auto arriveTime = seqgrp->IsPrefill() ? seqgrp->arriveTime : seqgrp->lastCompletionTime;
                auto waitTime = std::chrono::duration_cast<std::chrono::microseconds>(now - arriveTime);
                if (waitTime.count() >= 0) {
                    localResult->totalWaitTime += static_cast<uint64_t>(waitTime.count());
                }

                localResult->availableSeqCount++;
                localResult->waitBatchesCount += CountSequenceGroupSize(seqgrp);
                localResult->waitTokensCount += CountTokens(seqgrp, status);
                localResult->waitBlockNum += CountBlocks(seqgrp, status);
            },
            schedulerConfig_->maxBatchSize);
        return localResult;
    }

   private:
    std::shared_ptr<SchedulerConfig> schedulerConfig_;
    BlockSpaceManagerSPtr blockManager_;  // kv cache manager

    std::pair<size_t, size_t> GetNumComputeNewUnCachedAndCachedTokens(const SequenceGroupSPtr seqGroup,
                                                                      const SequenceStatus status) const;

    size_t CountSequenceGroupSize(SequenceGroupSPtr &seqgrp) const;

    size_t CountTokens(SequenceGroupSPtr &seqgrp, SequenceStatus status) const;

    size_t CountRunningTokens(SequenceGroupSPtr &seqgrp) const;

    size_t GetNumRequiredBlocks(size_t seqLen, size_t blockSize) const;

    // 统计给定序列组所需的块（block）数量，根据序列状态采用不同的计算方式。
    size_t CountBlocks(SequenceGroupSPtr &seqgrp, SequenceStatus status) const;
};
}  // namespace mindie_llm

#endif
