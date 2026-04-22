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

#include "event_dispatcher.h"

#include <condition_variable>

#include "endpoint_def.h"
#include "infer_instances.h"
#include "log.h"

using namespace mindie_llm;
const uint32_t CV_WAIT_TIME = 600;

EventDispatcher::EventDispatcher(const uint32_t logId, const std::string requestId, uint64_t timeout) noexcept
    : logId{logId}, requestId_{requestId}, timeout_{timeout} {}

void EventDispatcher::WaitEvent(httplib::DataSink *sink) {
    clearCount_.fetch_add(1);
    {
        boost::unique_lock<boost::mutex> lk(lock);
        auto ret = cv_.wait_until(lk, boost::chrono::steady_clock::now() + boost::chrono::seconds(timeout_));
        if (ret == boost::cv_status::timeout) {
            std::string message = "Engine callback timeout.";
            RequestIdNew requestId{requestId_};
            Status status = GetInferInstance()->ControlRequest(requestId, OperationV2::STOP);
            if (status.StatusCode() != Error::Code::OK) {
                ULOG_WARN(SUBMODLE_NAME_ENDPOINT,
                          GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_SINGLE_INFERENCE, TIMEOUT_WARNING),
                          "Failed stop inference. requestId: " << logId);
            } else {
                ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Stop inference success. RequestId is " << logId);
            }
            sink->write(message.data(), message.size());
            sink->done();
            isFinish_.store(true);
            sendCount_.store(0);
        } else {
            sink->write(message_.data(), message_.size());
            ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT,
                       "Wait event finish, load result is" << isFinish_.load() << ", logId is " << logId);
            if (isFinish_.load()) {
                sink->done();
            }
            sendCount_.store(0);
        }
    }
    clearCount_.fetch_sub(1);
}
void EventDispatcher::SendEvent(const std::string &message, bool finishFlag, std::string lastMessage) {
    if (isFinish_.load()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Timeout, but still callback. logId is " << logId);
        return;
    }
    {
        boost::unique_lock<boost::mutex> lk(lock);
        isFinish_.store(finishFlag);
        if (sendCount_.load() > 0 && !message_.empty()) {
            message_.append(message);
        } else {
            message_ = message;
        }
        sendCount_.fetch_add(1);
        if (!lastMessage.empty()) {
            message_.append(lastMessage);
        }
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Send event finish, finishFlag is " << finishFlag << ", logId is " << logId);
        cv_.notify_all();
    }
}

void EventDispatcher::Clear() {
    while (clearCount_.load() > 0) {
        std::this_thread::yield();
    }
}
