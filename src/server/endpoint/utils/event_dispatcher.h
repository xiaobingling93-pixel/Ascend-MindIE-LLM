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

#ifndef MIES_EVENT_DISPATCHER_H
#define MIES_EVENT_DISPATCHER_H

#include <boost/chrono.hpp>
#include <boost/thread.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>

#include "httplib.h"

namespace mindie_llm {
class EventDispatcher {
   public:
    EventDispatcher() = default;
    EventDispatcher(const uint32_t logId, const std::string requestId, uint64_t timeout) noexcept;
    ~EventDispatcher() = default;

    void WaitEvent(httplib::DataSink *sink);
    void SendEvent(const std::string &message, bool finishFlag, std::string lastMessage = "");
    void Clear();
    uint32_t logId{0};
    std::string requestId_;

   private:
    boost::mutex lock;
    boost::condition_variable cv_;
    std::string message_;
    std::string lastMessage_;
    std::atomic_bool isFinish_{false};
    std::atomic_int sendCount_{0};
    std::atomic_int clearCount_{0};
    uint64_t timeout_ = 600;
};
}  // namespace mindie_llm

#endif  // MIES_EVENT_DISPATCHER_H
