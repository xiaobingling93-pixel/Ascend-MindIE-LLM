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

#include "batch_latency_tracker.h"

namespace mindie_llm {
void BatchLatencyTracker::AddDataPoint(uint64_t value) {
    // delete oldest value
    if (queue_.size() == windowSize_) {
        queue_.pop_front();
    }

    // add new value
    queue_.push_back(value);
}
double BatchLatencyTracker::GetRecentAvgLatency(size_t forwardNum) {
    // 最小样本比例，避免前期样本太少计算的结果不对
    float minScale = 0.1;
    if (queue_.size() < (forwardNum * minScale)) {
        return 0;
    }

    size_t sumVal = 0;
    size_t count = (queue_.size() < forwardNum) ? queue_.size() : forwardNum;
    // 使用反向迭代器遍历队列最后count个元素
    auto rit = queue_.rbegin();
    for (size_t i = 0; i < count; ++i) {
        sumVal += *rit;
        ++rit;
    }

    return static_cast<double>(sumVal) / count;
}
}  // namespace mindie_llm
