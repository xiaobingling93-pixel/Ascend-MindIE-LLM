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

#pragma once

#include <limits>
#include <queue>
#include <unordered_map>

#include "evictor.h"

namespace mindie_llm {

const int LRU_EVICTOR_CLEANUP_THRESHOLD = 50;

class LRUEvictor : public Evictor {
   public:
    bool ContainsBlock(BlockId blockId) const override;

    EvictionResult Evict() override;

    void Add(BlockId blockId, HashValue prefixHash, size_t numHashedTokens, TimeStamp lastAccessedTime) override;

    void Update(BlockId blockId, TimeStamp lastAccessed) override;

    void Remove(BlockId blockId) override;

    size_t GetNumblocks() const override;

   private:
    struct BlockMetaData {
        BlockId blockId;
        HashValue prefixHash;
        size_t hashedTokensNum;
        TimeStamp lastAccessedTime;

        BlockMetaData() {}
        BlockMetaData(BlockId id, HashValue prefixHash, size_t hashedTokensNum, TimeStamp lastAccessed)
            : blockId(id), prefixHash(prefixHash), hashedTokensNum(hashedTokensNum), lastAccessedTime(lastAccessed) {}

        bool operator<(const BlockMetaData &other) const {
            if (std::abs(lastAccessedTime - other.lastAccessedTime) > std::numeric_limits<TimeStamp>::epsilon()) {
                return lastAccessedTime > other.lastAccessedTime;
            }

            return hashedTokensNum < other.hashedTokensNum;
        }
    };

   private:
    std::unordered_map<BlockId, BlockMetaData> candidates_;
    std::priority_queue<BlockMetaData> priorityQueue_;

   private:
    void Cleanup();
};

}  // namespace mindie_llm
