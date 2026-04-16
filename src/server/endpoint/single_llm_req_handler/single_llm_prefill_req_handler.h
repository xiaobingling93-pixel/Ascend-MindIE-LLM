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

#ifndef PREFILL_WRAPPER_H
#define PREFILL_WRAPPER_H

#include <limits>
#include <shared_mutex>
#include <vector>

#include "endpoint_def.h"
#include "grpc_communication_mng.h"
#include "http_rest_resource.h"
#include "infer_param.h"
#include "request_response/response.h"
#include "single_llm_req_handler_base.h"

namespace mindie_llm {
class SingleLLMPrefillReqHandler : public SingleLLMReqHandlerBase,
                                   public std::enable_shared_from_this<SingleLLMPrefillReqHandler> {
   public:
    SingleLLMPrefillReqHandler(ReqCtxPtr& ctx, uint16_t msgType, bool isRecompute);
    ~SingleLLMPrefillReqHandler() override = default;
    void Process(RequestSPtr request, const std::string& requestId, const uint64_t& timestamp) override;
    bool GetContextJsonBody(nlohmann::ordered_json& body) override;
    bool GetContextRequestId(std::string& requestId) override;
    void UpdateInferRequest(const std::vector<int64_t>& reqTokens, const int64_t& tokenLen,
                            RequestSPtr request) override;
    void UpdateInferParam(RequestSPtr request, const InferParamSPtr& inferParam) override;
    void SetBackManagerCallBack(RequestSPtr request) override;
    void SendResponseInfo(int code, const std::string& responseStr, bool needMetricsCollect = true) override;
    void SendResponse(int code, const std::string& responseStr) override;
    void SendResponseStream(bool isEnd, const std::string& responseStr) override;

    void BuildDecodeParameters(ResponseSPtr response, prefillAndDecodeCommunication::DecodeParameters& params);
    void BuildSamplingParametersFirst(prefillAndDecodeCommunication::DecodeParameters& params);
    void BuildSamplingParametersNext(prefillAndDecodeCommunication::DecodeParameters& params);
    void BuildInferParameters(prefillAndDecodeCommunication::DecodeParameters& params);
    void BuildMetricsParameters(prefillAndDecodeCommunication::DecodeParameters& params);
    bool GenerateFirstToken(ResponseSPtr response, bool expect);

   private:
    std::vector<int64_t> inputTokens_;  // reqTokens
    size_t oriReqTokenLen = std::numeric_limits<size_t>::max();
    std::vector<int64_t> respTokens_;
    bool AtomicReadWriteFinish(bool& expect);
    void GetPNodeAddr(bool containPort);
    void GetSingleLLMPrefillReqHandlerId();
    void ProcessResponseStream(bool isEnd);

    std::vector<int64_t> firstToken_;
    std::string reqId_;
    std::string tritonReqId_;
    std::string modelName_;
    std::string prefillNodeAddr_;
    // pInstanceId_ will be updated by GetSingleLLMPrefillReqHandlerId
    uint32_t pInstanceId_{0};
    uint16_t msgType_{MSG_TYPE_INVALID};
    RespBodyQueue respStr_;
    // 多个response并发call back使用;
    std::shared_mutex prefillCbMutex;
};
}  // namespace mindie_llm

#endif  // PREFILL_WRAPPER_H

// P节点的通信包含：
// 1. 与Coordinator节点通过context进行通信
// 2. 与D节点通过grpc通信

// P响应好像是可以推出多个Token，但是只响应一次。
// P节点的错误是否需要返回
