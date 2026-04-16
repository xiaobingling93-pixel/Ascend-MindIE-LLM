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

#ifndef REF_COUNTER_PROTOCOL_H
#define REF_COUNTER_PROTOCOL_H

#include <memory>
#include <vector>

#include "basic_types.h"

namespace mindie_llm {
class RefCounterProtocol {
   public:
    virtual ~RefCounterProtocol() = default;

    virtual RefCount Increase(BlockId blockId) = 0;

    virtual RefCount Decrease(BlockId blockId) = 0;

    virtual RefCount GetRefCount(BlockId blockId) const = 0;
};

using RefCounterProtocolSPtr = std::shared_ptr<RefCounterProtocol>;

RefCounterProtocolSPtr MakeRefCounterProtocol(const std::vector<BlockId> &allBlockIndices);

}  // namespace mindie_llm

#endif
