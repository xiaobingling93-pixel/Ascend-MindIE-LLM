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

#ifndef MIES_DRESULT_EVENT_DISPATCHER_H
#define MIES_DRESULT_EVENT_DISPATCHER_H

#include <boost/atomic.hpp>
#include <boost/lockfree/queue.hpp>
#include <queue>

#include "event_dispatcher.h"
#include "httplib.h"
namespace mindie_llm {
struct DResultWrapParam {
    std::string body;
    std::string prefix;
    std::string reqId;
    std::string tritonReqId;
};

class DResultEventDispatcher {
   public:
    explicit DResultEventDispatcher();
    ~DResultEventDispatcher();

    void WaitEvent(httplib::DataSink *sink);
    void SendEvent(const std::string &message, bool finishFlag, std::string reqInfo);
    static void WrapChunkedDResponse(std::string &msg, const DResultWrapParam &param);
    boost::chrono::nanoseconds GetIntervalFromPrevSend() const;

   private:
    void ClearQueue();
    void WriteStreamMessage(httplib::DataSink *sink);
    std::mutex qMutex_;
    std::condition_variable cv_;
    std::atomic_bool isFinish_{false};
    std::atomic_bool isDestroyed_{false};
    boost::chrono::time_point<boost::chrono::steady_clock> lastTimestamp_{boost::chrono::steady_clock::now()};
    boost::lockfree::queue<std::string *> queue_{10000};
};
}  // namespace mindie_llm

#endif
