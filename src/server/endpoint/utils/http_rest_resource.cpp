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

#define CPPHTTPLIB_OPENSSL_SUPPORT

#include "http_rest_resource.h"

#include <sstream>

#include "endpoint_def.h"
#include "http_handler.h"
#include "log.h"
#include "nlohmann/json.hpp"
#include "prometheus_metrics.h"

#ifdef UT_ENABLED
#define LOCAL_API
#else
#define LOCAL_API static
#endif

using OrderedJson = nlohmann::ordered_json;

namespace mindie_llm {
const std::string RequestContext::requestIdPrefix{"request_context_"};
std::atomic<uint64_t> RequestContext::requestIdContext(1UL);

RequestContext::RequestContext(const httplib::Request &request, httplib::Response &response) noexcept
    : req_(request), res_(response) {
    method = request.method;
    path = request.path;
    version = request.version;
    msgBody = request.body;
    cbId = requestIdContext++;
    inferRequestId = cbId;
    reqUuid = response.get_header_value("RequestUUID");
    if (request.is_connection_closed != nullptr) {
        isConnectionClosed_ = request.is_connection_closed;
    }
}

void RequestContext::SetResponseFinished(bool isFinished) {
    std::lock_guard<std::mutex> lock(streamMutex);
    isStreamResponseFinished = isFinished;
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "[ThreadPoolMonitor] SetResponseFinished for RequestContext " << this);
}

std::unique_lock<std::mutex> RequestContext::LockAndCheckResponseFinished(bool &flagOut) {
    // return a RAII lock for atomic operation
    std::unique_lock<std::mutex> lock(streamMutex);
    flagOut = isStreamResponseFinished;
    return lock;
}

InferRequestIdType RequestContext::InferRequestId() const { return inferRequestId; }

void RequestContext::SetStopInferFunction(std::function<void()> handler) { disconnectHandler_ = std::move(handler); }

void RequestContext::StopInferRequest() {
    if (disconnectHandler_) {
        disconnectHandler_();
    } else {
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT,
                  GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_SERVER_REQUEST, LOCAL_INVOKING_ERROR),
                  "Function 'disconnectHandler_' is not set. The request may have been completed.");
    }
}

uint32_t RequestContext::CallbackId() const noexcept { return cbId; }

void RequestContext::SetCallbackId(uint32_t id) noexcept { cbId = id; }

void RequestContext::SetHTTPRequestUUID(std::string uuid) { reqUuid = uuid; };

std::string RequestContext::GetHTTPRequestUUID() { return reqUuid; };

const std::string &RequestContext::Method() const noexcept { return method; }

const std::string &RequestContext::Version() const noexcept { return version; }

const std::string &RequestContext::Path() const noexcept { return path; }

const std::string &RequestContext::MsgBody() const noexcept { return msgBody; }

const httplib::Request &RequestContext::Req() const noexcept { return req_; }

httplib::Response &RequestContext::Res() noexcept { return res_; }

bool RequestContext::IsConnectionClosed() const noexcept { return isConnectionClosed_(); }

int HttpRestResource::ResponseNobody(const ReqCtxPtr &requestContext, int code) noexcept {
    return ResponseWithBody(requestContext, code, "", "");
}

int HttpRestResource::ResponseJsonBody(const ReqCtxPtr &requestContext, int code, const std::string &body) noexcept {
    return ResponseWithBody(requestContext, code, "application/json", body);
}

int HttpRestResource::ResponseWithBody(const ReqCtxPtr &requestContext, int code, const std::string &contentType,
                                       const std::string &body,
                                       const std::shared_ptr<std::map<std::string, std::string>> &headers) noexcept {
    // 设置响应状态码
    auto &response = requestContext->Res();
    response.status = code;

    // 设置响应头部
    response.set_header("Content-Type", contentType);
    response.set_header("Connection", "close");
    if (headers != nullptr) {
        for (auto &it : *headers) {
            response.set_header(it.first, it.second);
        }
    }
    // 直接发送响应体
    response.set_content(body, contentType);
    requestContext->SetResponseFinished(true);
    return 0;
}

std::string HttpRestResource::WrapperJson(const std::string &error, const std::string &errorType) noexcept {
    OrderedJson jsonObj;
    jsonObj["error"] = error;
    jsonObj["error_type"] = errorType;
    return jsonObj.dump(-1, ' ', false, OrderedJson::error_handler_t::replace);
}

std::string HttpRestResource::WrapperStatusJson(const std::string &message) noexcept {
    OrderedJson jsonObj;
    jsonObj["status"] = message;
    return jsonObj.dump(-1, ' ', false, OrderedJson::error_handler_t::replace);
}

void HttpRestResource::ResponseChunkedBody(const ReqCtxPtr &requestContext,
                                           std::shared_ptr<EventDispatcher> &eventDispatcher) noexcept {
    auto &response = requestContext->Res();
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "HttpRestResource::set_chunked_content_provider");
    response.set_chunked_content_provider(
        "text/event-stream", [&, eventDispatcher](size_t /* offset */, httplib::DataSink &sink) {
            eventDispatcher->WaitEvent(&sink);
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "HttpRestResource::Stream Response Finished");
            return true;
        });
}
}  // namespace mindie_llm
