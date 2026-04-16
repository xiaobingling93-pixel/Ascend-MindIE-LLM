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

#ifndef EVICTOR_H
#define EVICTOR_H

#include <memory>
#include <vector>

#include "basic_types.h"

namespace mindie_llm {
struct EvictionResult {
    HashValue prefixHash;
    BlockId blockId;
};

class Evictor {
   public:
    Evictor() = default;

    virtual ~Evictor() = default;

    virtual bool ContainsBlock(BlockId blockId) const = 0;

    virtual EvictionResult Evict() = 0;

    virtual void Add(BlockId blockId, HashValue prefixHash, size_t numHashedTokens, TimeStamp lastAccessedTime) = 0;

    virtual void Update(BlockId blockId, TimeStamp lastAccessed) = 0;

    virtual void Remove(BlockId blockId) = 0;

    virtual size_t GetNumblocks() const = 0;
};

enum class EvictionPolicy { LRU };

using EvictorPtr = std::unique_ptr<Evictor>;

EvictorPtr MakeEvictor(EvictionPolicy policy = EvictionPolicy::LRU);
}  // namespace mindie_llm

#endif
