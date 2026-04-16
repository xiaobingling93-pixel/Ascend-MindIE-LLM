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

#ifndef COMMON_WRAPPER_H
#define COMMON_WRAPPER_H

#include <boost/chrono.hpp>
#include <boost/thread.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>

#include "http_rest_resource.h"
#include "single_llm_req_handler_base.h"

using BoostTimePoint = boost::chrono::steady_clock::time_point;

namespace mindie_llm {
class SingleLLMPnDReqHandler : public SingleLLMReqHandlerBase,
                               public std::enable_shared_from_this<SingleLLMPnDReqHandler> {
   public:
    using SingleLLMReqHandlerBase::GetContextJsonBody;
    explicit SingleLLMPnDReqHandler(ReqCtxPtr &ctx, bool isFlexLocalReq = false);
    ~SingleLLMPnDReqHandler() override;

    void Process(RequestSPtr request, const std::string &inputId, const uint64_t &timestamp = 0) override;
    bool GetContextJsonBody(nlohmann::ordered_json &body) override;
    bool GetContextRequestId(std::string &requestId) override;
    void UpdateInferRequest(const std::vector<int64_t> &reqTokens, const int64_t &oriReqTokenLen,
                            RequestSPtr request) override;
    void UpdateInferParam(RequestSPtr request, const InferParamSPtr &inferParam) override;
    void SetBackManagerCallBack(RequestSPtr request) override;
    void SendResponseInfo(int code, const std::string &responseStr, bool needMetricsCollect = true) override;
    void SendResponse(int code, const std::string &responseStr) override;
    void SendResponseStream(bool isEnd, const std::string &responseStr) override;

   private:
    // 处理非流式的请求，在此函数中完成本次请求的全部处理，并响应http客户端
    void ProcessNonStreamModeRequest(const std::string &inputId, const uint64_t &timestamp = 0);

    bool ProcessNonStreamBatchResponse(const std::vector<ResponseSPtr> &responseImpls);

    // 处理非流式请求的一次来自引擎层的响应，返回值表示是否结束，结束时在内部自动响应http客户端
    bool ProcessOneNonStreamResponse(const ResponseSPtr &response);

    // 处理流式请求，在此函数完成http响应头回复，数据部分注册到httplib拉取chunk
    void ProcessStreamModeRequest(const std::string &inputId, const uint64_t &timestamp = 0);

    // 注册给httplib拉起流式chunk响应数据的函数，返回值表示产生数据长度
    void StreamResponseFillHttpSink(const std::string &inputId, httplib::DataSink &sink, const uint64_t &timestamp = 0);

    // 处理流式请求中，一次来自引擎层的响应，返回值表示是否结束
    bool ProcessOneStreamResponse(const ResponseSPtr &response, RespBodyQueue &text);

    void ProcessOneResponsePrometheusMetrics(const ResponseSPtr &response);

    /// Get inference timeout point and its name.
    /// \param tokenStartTime    [in]  Token start time.
    /// \param lastTimePoint     [out] Inference timeout point.
    /// \param lastTimePointName [out] Inference time point description.
    void GetTimeoutPoint(const BoostTimePoint &tokenStartTime, BoostTimePoint &lastTimePoint, std::string &desc);
    int64_t GetElapsedTimeMillis(const std::any &startTimePoint) const;

    // 处理flex节点到flex节点的请求，添加标志位
    bool isFlexLocalReq_{false};
};
}  // namespace mindie_llm
#endif  // COMMON_WRAPPER_H
