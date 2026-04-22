/*
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
#include "qps_tracker.h"

#include <chrono>

using namespace mindie_llm;
using namespace std;
inline int64_t NowInMilli() {
    const auto now = chrono::steady_clock::now();
    return chrono::time_point_cast<chrono::milliseconds>(now).time_since_epoch().count();
}

QPSTracker::QPSTracker(int timeWindowInterval, int bucketTimeInterval)
    : bucketTimeInterval_(bucketTimeInterval),
      timeWindowInterval_(timeWindowInterval),
      bucketsNum_(timeWindowInterval / bucketTimeInterval),
      buckets_(bucketsNum_) {}
float QPSTracker::GetQPS() {
    int64_t nowInMs = NowInMilli();
    int64_t endBucketId = static_cast<int64_t>(nowInMs / bucketTimeInterval_);
    int64_t beginBucketId = endBucketId - static_cast<int64_t>(bucketsNum_) + 1;
    int64_t requestNum = 0;
    for (size_t i = 0; i < buckets_.size(); i++) {
        if (buckets_[i].bucketId >= beginBucketId && buckets_[i].bucketId <= endBucketId) {
            requestNum += buckets_[i].counter.load(memory_order_relaxed);
        }
    }
    // timeWindowInterval_ is in milli seconds
    const float msPerSecond = 1000;
    return (static_cast<float>(requestNum)) / static_cast<float>(timeWindowInterval_) * msPerSecond;
}

void QPSTracker::Record() {
    int64_t nowInMs = NowInMilli();
    int64_t bucketId = nowInMs / bucketTimeInterval_;
    QCntBucket &bucket = buckets_[static_cast<size_t>(bucketId) % bucketsNum_];
    if (bucket.bucketId == bucketId) {
        bucket.counter.fetch_add(1, memory_order_relaxed);
    } else {  // not accurate when there is high concurrrency, some requests will not be calcuated
        bucket.counter.store(1, memory_order_relaxed);
        bucket.bucketId = bucketId;
    }
}
