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

#include "ref_counter.h"

#include <stdexcept>

namespace mindie_llm {

RefCounterProtocolSPtr MakeRefCounterProtocol(const std::vector<BlockId> &allBlockIndices) {
    return std::make_shared<RefCounter>(allBlockIndices);
}

RefCounter::RefCounter(const std::vector<BlockId> &allBlockIndices) {
    for (BlockId id : allBlockIndices) {
        refCounts_[id] = 0;
    }
}

RefCount RefCounter::Increase(BlockId blockId) {
    if (refCounts_.count(blockId) == 0 || refCounts_[blockId] < 0) {
        throw std::runtime_error("the blockId's information invalid.");
    }

    return ++refCounts_[blockId];
}

RefCount RefCounter::Decrease(BlockId blockId) {
    if (refCounts_.count(blockId) == 0 || refCounts_[blockId] <= 0) {
        throw std::runtime_error("the blockId's information invalid.");
    }

    return --refCounts_[blockId];
}

RefCount RefCounter::GetRefCount(BlockId blockId) const {
    if (refCounts_.count(blockId) == 0) {
        throw std::runtime_error("the blockId's information invalid.");
    }

    return refCounts_.at(blockId);
}
}  // namespace mindie_llm
