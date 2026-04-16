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

#include "dresult_event_dispatcher.h"

#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <stdexcept>

#include "endpoint_def.h"
#include "log.h"
#include "memory_utils.h"

namespace mindie_llm {
DResultEventDispatcher::DResultEventDispatcher() {}

DResultEventDispatcher::~DResultEventDispatcher() {
    isDestroyed_ = true;
    ClearQueue();
}

void DResultEventDispatcher::ClearQueue() {
    std::string *temp;
    while (queue_.pop(temp)) {
        delete temp;
    }
}

void DResultEventDispatcher::WriteStreamMessage(httplib::DataSink *sink) {
    std::string *msg;
    while (!queue_.empty()) {
        while (!queue_.pop(msg)) {
        }
        if (sink && msg) {
            sink->write(msg->data(), msg->size());
        }
        delete msg;
    }
    if (isFinish_.load()) {
        sink->done();
    }
}

void DResultEventDispatcher::WaitEvent(httplib::DataSink *sink) {
    while (!isDestroyed_.load()) {
        std::unique_lock<std::mutex> lock(qMutex_);
        cv_.wait(lock, [this]() { return isDestroyed_.load() || !queue_.empty(); });
        if (isDestroyed_.load()) {
            return;
        }
        try {
            WriteStreamMessage(sink);
        } catch (const std::exception &e) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, SYSTEM_INVOKING_ERROR),
                       "Failed to write message into stream : " << e.what());
            ClearQueue();
        } catch (...) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, SYSTEM_INVOKING_ERROR),
                       "Unknown exception caught");
            ClearQueue();
        }
    }
}

void DResultEventDispatcher::SendEvent(const std::string &message, bool finishFlag, std::string reqInfo) {
    if (isFinish_.load()) {
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_SPLITWISE, CHECK_ERROR),
                  "D Result long connection is closed, won't send data, Request Id is" << reqInfo);
        return;
    }
    lastTimestamp_ = boost::chrono::steady_clock::now();
    isFinish_.store(finishFlag);
    std::string *msg = new std::string(message);
    while (!queue_.push(msg)) {
    }
    cv_.notify_one();
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT,
               "Send event finish, finishFlag is " << finishFlag << " request info is " << reqInfo);
}

/**
 * WrapChunkedDResponse: 将需要返回给Coordinator的body按D响应的格式包装成二进制
 * [OUT] msg: 包装后的二进制
 * [IN]  param: 基本参数
 * D响应的格式：
 *      reqId:<requestID> + \0 + <body> + \0
 *      <body>由前缀 + 内容组成
 * 例：假设token的内容为 data:{xxx} ，requestId为 1234567
 *     普通报文(由于前缀是data:，token的内容的开头也有一个data:，因而这里会有2个data:)
 *      reqId:1234567\0data:data:{xxx}\0
 *     最后一个报文，前缀为lastData
 *      reqId:1234567\0lastData:data:{xxx}\0
 */
void DResultEventDispatcher::WrapChunkedDResponse(std::string &msg, const DResultWrapParam &param) {
    auto reqId = param.tritonReqId.empty() ? param.reqId : param.tritonReqId;
    std::string idPrefix = "reqId:";
    std::string bodyPrefix = param.prefix;
    char separator = '\0';
    size_t separatorSize = 1;
    try {
        size_t len = 0;
        auto safeAdd = [&len](size_t value_to_add) -> bool {
            if (value_to_add > std::numeric_limits<size_t>::max() - len) {
                return false;
            }
            len += value_to_add;
            return true;
        };
        if (!safeAdd(idPrefix.size()) || !safeAdd(reqId.size()) || !safeAdd(separatorSize) ||
            !safeAdd(bodyPrefix.size()) || !safeAdd(param.body.size()) || !safeAdd(separatorSize)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, ABNORMAL_TRANSMISSION_ERROR),
                       "Message size would overflow size_t. RequestID is " << reqId);
            return;
        }
        msg.reserve(len);
        msg.append(idPrefix)
            .append(reqId)
            .append(1, separator)
            .append(bodyPrefix)
            .append(param.body)
            .append(1, separator);
    } catch (const std::bad_alloc &e) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, ABNORMAL_TRANSMISSION_ERROR),
                   "Failed to alloc mem. requestId: " << reqId);
        return;
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, ABNORMAL_TRANSMISSION_ERROR),
                   "Failed to resize msg. requestId: " << reqId);
        return;
    }
}

boost::chrono::nanoseconds DResultEventDispatcher::GetIntervalFromPrevSend() const {
    return boost::chrono::steady_clock::now() - lastTimestamp_;
}
}  // namespace mindie_llm
