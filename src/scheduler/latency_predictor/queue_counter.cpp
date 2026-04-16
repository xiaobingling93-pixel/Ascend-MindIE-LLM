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

#include "latency_predictor/queue_counter.h"

#include <iostream>

namespace mindie_llm {
QueueCounter::QueueCounter(const std::shared_ptr<SchedulerConfig> &schedulerConfig,
                           const std::shared_ptr<BlockSpaceManager> &blockManager)
    : schedulerConfig_(schedulerConfig), blockManager_(blockManager) {}

std::pair<size_t, size_t> QueueCounter::GetNumComputeNewUnCachedAndCachedTokens(const SequenceGroupSPtr seqGroup,
                                                                                const SequenceStatus status) const {
    size_t numCachedNewTokens = 0;
    size_t numUncachedNewTokens = 0;

    std::vector<SequenceSPtr> seqs = seqGroup->GetSequences(status);
    for (SequenceSPtr &seq : seqs) {
        if (!seq->IsPrefill()) {
            numUncachedNewTokens += 1;
            continue;
        }

        const size_t numComputedTokenSeq = seq->GetNumComputedTokens();
        Assert(seq->GetLen() >= numComputedTokenSeq);
        const size_t allNumNewTokensSeq = seq->GetLen() - numComputedTokenSeq;
        if (!schedulerConfig_->enablePrefixCache) {
            numUncachedNewTokens += allNumNewTokensSeq;
            continue;
        }

        const size_t numCachedTokensSeq = blockManager_->GetSeqNumCachedTokens(seq);
        const size_t numCachedNewTokensSeq =
            numCachedTokensSeq >= numComputedTokenSeq ? numCachedTokensSeq - numComputedTokenSeq : 0;
        const size_t numUncachedNewTokensSeq = allNumNewTokensSeq - numCachedNewTokensSeq;
        numUncachedNewTokens += numUncachedNewTokensSeq;
        numCachedNewTokens += numCachedNewTokensSeq;
    }

    if (numUncachedNewTokens == 0 && numCachedNewTokens > 0) {
        numUncachedNewTokens = 1;
        numCachedNewTokens -= 1;
    }

    return {numUncachedNewTokens, numCachedNewTokens};
}

size_t QueueCounter::CountSequenceGroupSize(SequenceGroupSPtr &seqgrp) const {
    size_t count = 1;  // 包含root序列组
    if (seqgrp->sampling && seqgrp->sampling->enableParallelSampling) {
        count += seqgrp->seqId2ParallelSeqGroup_.Size();
    }
    return count;
}

size_t QueueCounter::CountTokens(SequenceGroupSPtr &seqgrp, SequenceStatus status) const {
    if (status == SequenceStatus::RUNNING) {
        return CountRunningTokens(seqgrp);
    }
    auto [numNewTokensUncached, _] = GetNumComputeNewUnCachedAndCachedTokens(seqgrp, status);
    static_cast<void>(_);
    return numNewTokensUncached;
}

size_t QueueCounter::CountRunningTokens(SequenceGroupSPtr &seqgrp) const {
    if (seqgrp->seqs_.empty()) {
        return 0;
    }

    size_t count = seqgrp->seqs_[0]->data_.outputTokenIds.size();
    if (seqgrp->sampling && seqgrp->sampling->enableParallelSampling) {
        std::vector<SequenceId> parallelSeqIds = seqgrp->seqId2ParallelSeqGroup_.KeySet();
        for (auto &seqId : parallelSeqIds) {
            auto parallelSeqGrp = seqgrp->seqId2ParallelSeqGroup_.Get(seqId);
            if (parallelSeqGrp.has_value()) {
                count += parallelSeqGrp.value()->firstSeq->data_.outputTokenIds.size();
            }
        }
    }
    return count;
}

size_t QueueCounter::GetNumRequiredBlocks(size_t seqLen, size_t blockSize) const {
    if (blockSize == 0) {
        throw std::runtime_error("the blockSize should not be zero");
    }

    return (seqLen + blockSize - 1) / blockSize;
}

// 统计给定序列组所需的块（block）数量，根据序列状态采用不同的计算方式。
size_t QueueCounter::CountBlocks(SequenceGroupSPtr &seqgrp, SequenceStatus status) const {
    // 情况1：如果是RUNNING状态且序列组非空
    if (status == SequenceStatus::RUNNING && !seqgrp->seqs_.empty()) {
        // 获取第一个序列的ID
        SequenceId seqId = seqgrp->seqs_[0]->seqId_;
        // 直接查询块管理器获取该序列实际使用的块数量
        const auto allIds = blockManager_->GetBlockIds(seqId);
        if (allIds.empty()) {
            return 0;
        }
        return allIds[0].size();
    }

    // 情况2：其他状态（如WAITING、SWAPPED等），根据序列长度和块大小计算预估需要的块数量
    if (seqgrp->seqs_.empty()) {
        throw std::runtime_error("the sequenceGroup has no seq");
    }
    return GetNumRequiredBlocks(seqgrp->seqs_[0]->GetTokenIds().size(), schedulerConfig_->cacheBlockSize);
}
}  // namespace mindie_llm
