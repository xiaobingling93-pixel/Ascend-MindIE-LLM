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

#ifndef COPY_ON_WRITE_TRACKER_H
#define COPY_ON_WRITE_TRACKER_H

#include "ref_counter_protocol.h"

namespace mindie_llm {
// CopyOnWriteTracker maintains a reference counter for each block id, and records required block copy operation(src
// block id to dst block id) . And this operations need to be sent to executor to avoid memory write conflicts in this
// round of scheduling.
// Since only fully filled blocks can be shared and fully filled blocks will not be appended, this CowTracker is not
// used now.
class CopyOnWriteTracker {
   public:
    CopyOnWriteTracker() = default;

    explicit CopyOnWriteTracker(RefCounterProtocolSPtr &refCounterSPtr) : refCounterSPtr_(refCounterSPtr) {}

    bool IsAppendable(BlockId blockId) const;  // block id with refcount == 1 is appendable

    void RecordCow(BlockId srcBlockId, BlockId targetBlockId);

    std::vector<std::pair<BlockId, BlockId>> ClearCows();

   private:
    RefCounterProtocolSPtr refCounterSPtr_;
    std::vector<std::pair<BlockId, BlockId>> copyOnWrites_;
};
}  // namespace mindie_llm

#endif
