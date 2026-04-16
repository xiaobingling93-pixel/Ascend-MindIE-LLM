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

#include "copy_on_write_tracker.h"

#include <stdexcept>

namespace mindie_llm {
bool CopyOnWriteTracker::IsAppendable(BlockId blockId) const {
    RefCount refCount = refCounterSPtr_->GetRefCount(blockId);

    return refCount <= 1;
}

void CopyOnWriteTracker::RecordCow(BlockId srcBlockId, BlockId targetBlockId) {
    copyOnWrites_.emplace_back(std::make_pair(srcBlockId, targetBlockId));
}

std::vector<std::pair<BlockId, BlockId>> CopyOnWriteTracker::ClearCows() {
    std::vector<std::pair<BlockId, BlockId>> result(copyOnWrites_.begin(), copyOnWrites_.end());
    copyOnWrites_.clear();
    return result;
}
}  // namespace mindie_llm
