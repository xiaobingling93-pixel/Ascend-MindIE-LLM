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

#ifndef REF_COUNTER_H
#define REF_COUNTER_H

#include <unordered_map>

#include "ref_counter_protocol.h"

namespace mindie_llm {
class RefCounter : public RefCounterProtocol {
   public:
    explicit RefCounter(const std::vector<BlockId> &allBlockIndices);

    RefCount Increase(BlockId blockId) override;

    RefCount Decrease(BlockId blockId) override;

    RefCount GetRefCount(BlockId blockId) const override;

   private:
    std::unordered_map<BlockId, RefCount> refCounts_;
};
}  // namespace mindie_llm

#endif
