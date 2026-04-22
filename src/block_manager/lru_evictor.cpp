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

#include "lru_evictor.h"

#include <stdexcept>

#include "math_utils.h"

namespace mindie_llm {

bool LRUEvictor::ContainsBlock(BlockId blockId) const { return candidates_.find(blockId) != candidates_.end(); }

EvictionResult LRUEvictor::Evict() {
    if (candidates_.empty()) {
        throw std::runtime_error("No usable cache memory!");
    }

    while (!priorityQueue_.empty()) {
        BlockMetaData data = priorityQueue_.top();
        priorityQueue_.pop();

        auto it = candidates_.find(data.blockId);
        if (it != candidates_.end() && IsClose(it->second.lastAccessedTime, data.lastAccessedTime)) {
            HashValue prefixHash = it->second.prefixHash;
            candidates_.erase(it);
            return {prefixHash, data.blockId};
        }
    }

    throw std::runtime_error("No usable cache memory left");
}

void LRUEvictor::Add(BlockId blockId, HashValue prefixHash, size_t numHashedTokens, TimeStamp lastAccessedTime) {
    BlockMetaData data{blockId, prefixHash, numHashedTokens, lastAccessedTime};
    candidates_[blockId] = data;
    priorityQueue_.push(data);

    if (priorityQueue_.size() > LRU_EVICTOR_CLEANUP_THRESHOLD * candidates_.size()) {
        Cleanup();
    }
}

void LRUEvictor::Update(BlockId blockId, TimeStamp lastAccessed) {
    candidates_[blockId].lastAccessedTime = lastAccessed;
}

void LRUEvictor::Remove(BlockId blockId) {
    if (candidates_.find(blockId) != candidates_.end()) {
        candidates_.erase(blockId);
    }
}

size_t LRUEvictor::GetNumblocks() const { return candidates_.size(); }

void LRUEvictor::Cleanup() {
    std::priority_queue<BlockMetaData> queue;

    for (const auto &candidate : candidates_) {
        queue.push(candidate.second);
    }

    priorityQueue_ = std::move(queue);
}

EvictorPtr MakeEvictor(EvictionPolicy policy) {
    if (policy == EvictionPolicy::LRU) {
        return std::make_unique<LRUEvictor>();
    }

    throw std::runtime_error("Unknown cache eviction policy");
}
}  // namespace mindie_llm
