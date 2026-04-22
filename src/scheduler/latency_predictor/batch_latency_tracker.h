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

#ifndef BATCH_LATENCY_TRACKER
#define BATCH_LATENCY_TRACKER

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <numeric>
#include <set>
#include <vector>

namespace mindie_llm {
class BatchLatencyTracker {
   public:
    explicit BatchLatencyTracker(size_t windowSize = 1e6) : windowSize_(windowSize) {}

    void AddDataPoint(uint64_t value);

    double GetRecentAvgLatency(size_t forwardNum);

   private:
    size_t windowSize_;

    std::deque<uint64_t> queue_;
};
}  // namespace mindie_llm

#endif  // BATCH_LATENCY_TRACKER
