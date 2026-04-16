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

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

#include "http_rest_resource.h"
#include "httplib.h"

using InferRequestIdType = uint32_t;

namespace mindie_llm {
class ThreadPoolMonitor final : public httplib::TaskQueue {
   public:
    explicit ThreadPoolMonitor(size_t n, size_t mqr = 0);
    ~ThreadPoolMonitor() override;

    bool enqueue(std::function<void()> fn) override;
    void shutdown() override;

    void AddRequestToMonitor(std::shared_ptr<RequestContext> requestContext);
    void RemoveMonitorRequest(std::string reqid);
    void CheckAndRemoveClosedConnections();

   private:
    void Monitor();

    struct Worker {
        explicit Worker(ThreadPoolMonitor &pool) : pool_(pool) {}
        void operator()();
        std::function<void()> GetNextJob();
        ThreadPoolMonitor &pool_;
    };

    std::vector<std::thread> threads_;
    std::list<std::function<void()>> jobs_;
    bool shutdown_;
    size_t max_queued_requests_ = 0;

    std::condition_variable cond_;
    std::mutex mutex_;

    std::map<std::string, std::shared_ptr<RequestContext>> monitorRequests_;
    std::atomic<bool> stop_monitor_flag;
    std::thread monitor_thread;
    std::shared_mutex monitor_map_mutex_;
};
}  // namespace mindie_llm
