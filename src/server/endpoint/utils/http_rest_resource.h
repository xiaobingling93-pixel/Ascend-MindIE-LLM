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

#ifndef ENDPOINT_HTTP_REST_RESOURCE_H
#define ENDPOINT_HTTP_REST_RESOURCE_H

#include <sys/types.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <unordered_map>

#include "event_dispatcher.h"
#include "httplib.h"
#include "infer_instances.h"
#include "log.h"

namespace mindie_llm {
using InferRequestIdType = uint32_t;
using RequestDisconnectHandler = std::function<void>();

const std::unordered_map<uint64_t, std::string> g_exceptionInfo = {
    {httplib::StatusCode::BadRequest_400, "Bad Request"},
    {httplib::StatusCode::NotFound_404, "NotFoundError"},
    {httplib::StatusCode::UnprocessableContent_422, "Input Validation Error"},
    {httplib::StatusCode::FailedDependency_424, "Generation Error"},
    {httplib::StatusCode::InternalServerError_500, "Incomplete Generation"},
    {httplib::StatusCode::ServiceUnavailable_503, "Service Unavailable"}};
class RequestContext {
   public:
    RequestContext(const httplib::Request &request, httplib::Response &response) noexcept;
    ~RequestContext() = default;

   public:
    uint32_t CallbackId() const noexcept;
    void SetCallbackId(uint32_t id) noexcept;
    void SetResponseFinished(bool isFinished);
    std::unique_lock<std::mutex> LockAndCheckResponseFinished(bool &flagOut);
    InferRequestIdType InferRequestId() const;
    void SetHTTPRequestUUID(std::string uuid);
    std::string GetHTTPRequestUUID();
    void SetStopInferFunction(std::function<void()> handler);
    void StopInferRequest();
    const std::string &Method() const noexcept;
    const std::string &Version() const noexcept;
    const std::string &Path() const noexcept;
    const std::string &MsgBody() const noexcept;
    const httplib::Request &Req() const noexcept;
    httplib::Response &Res() noexcept;
    bool IsConnectionClosed() const noexcept;
    std::string ToString() const {
        std::ostringstream oss;
        oss << "RequestContext { " << "CallbackId: " << cbId << ", InferRequestId: " << inferRequestId
            << ", RequestUUID: " << reqUuid << ", Method: " << method << ", Version: " << version << ", Path: " << path
            << ", MsgBody: " << msgBody << " }";
        return oss.str();
    }

   private:
    static const std::string requestIdPrefix;
    static std::atomic<uint64_t> requestIdContext;

    uint64_t cbId = 0;
    uint32_t inferRequestId = -1;
    bool isStreamResponse{false};
    // for stream mode, client will receive response first, the data will transfer in multiple chunks
    // use this flag to check if stream response is completed
    bool isStreamResponseFinished{false};
    std::string reqUuid;
    std::string method;
    std::string version;
    std::string path;
    std::string msgBody;
    const httplib::Request &req_;
    httplib::Response &res_;

    std::function<void()> disconnectHandler_;
    std::function<bool()> isConnectionClosed_ = []() { return true; };
    mutable std::mutex streamMutex;
};

using ReqCtxPtr = std::shared_ptr<RequestContext>;

class HttpRestResource {
   public:
    static int ResponseNobody(const ReqCtxPtr &requestContext, int code) noexcept;
    static int ResponseJsonBody(const ReqCtxPtr &requestContext, int code, const std::string &body) noexcept;
    static int ResponseWithBody(const ReqCtxPtr &requestContext, int code, const std::string &contentType,
                                const std::string &body,
                                const std::shared_ptr<std::map<std::string, std::string>> &headers = nullptr) noexcept;
    static void ResponseChunkedBody(const ReqCtxPtr &requestContext,
                                    std::shared_ptr<EventDispatcher> &eventDispatcher) noexcept;
    static std::string WrapperJson(const std::string &error, const std::string &errorType = "validation") noexcept;
    static std::string WrapperStatusJson(const std::string &message) noexcept;
};
}  // namespace mindie_llm

#endif  // ENDPOINT_HTTP_REST_RESOURCE_H
