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

#ifndef _DUMMY_QUOTA_MANAGER
#define _DUMMY_QUOTA_MANAGER
#include <atomic>
#include <boost/thread/sync_queue.hpp>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "process_group.h"

namespace mindie_llm {
using namespace std::chrono;
constexpr size_t REQ_WAIT_LOG_LOOP = 3000;
constexpr int INIT_QUOTA_NUM = 10000;
// dummyBatchQueue_触发降级为无脑dummy的队列元素数量
constexpr int MAX_ALLOWED_DUMMY_QUEUE_PENDINGS = 1000;
class DummyQuotaManager {
   public:
    explicit DummyQuotaManager(int rank = 0, int quota = INIT_QUOTA_NUM);
    ~DummyQuotaManager();
    bool AcquireQuota(bool isDummy);
    void Wakeup();

   private:
    void AcrossProcSyncThreadEntry_();
    void AllGather_(bool iAmDummy, bool &allDummy, int &maxQuota, bool dummyBatchSync);
    boost::sync_queue<bool> dummyBatchQueue_;  // blocking queue for dummy/real Batch
    int initQuota_{INIT_QUOTA_NUM};
    std::atomic<int> quotaLeft_{INIT_QUOTA_NUM};
    std::atomic<bool> threadNeedStop_{false};
    std::thread acrossProcSyncThread_;
    void QuotaAllGather_(bool &iAmDummy, bool &allDummy, int &maxQuota);

    // real request blocking queue for wakeup(fill to full quota) purpose
    boost::sync_queue<bool> reqCommingQueue_;
    void WakeupAllGather_(bool &iAmDummy, bool &allDummy, int &maxQuota);
    int rank_;  // for debugging purpose

    std::mutex notifyMutex_;
    std::condition_variable notifyCv_;  // combine with notifyMutex_ to notify quota is restored
    void WaitForQuota_();               // blocking
    void NotifyQuotaAvailable_();
    bool hasRequest{false};
};

using DummyQuotaManagerSPtr = std::shared_ptr<DummyQuotaManager>;
}  // namespace mindie_llm
#endif
