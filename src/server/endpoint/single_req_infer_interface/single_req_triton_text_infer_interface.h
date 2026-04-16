/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef ENDPOINT_TRITON_TEXT_INFER_H
#define ENDPOINT_TRITON_TEXT_INFER_H

#include <cstdint>
#include <vector>

#include "httplib.h"
#include "infer_param.h"
#include "single_req_infer_interface_base.h"

namespace mindie_llm {
/**
 * @brief Triton text 格式的推理请求处理类
 */
class SingleReqTritonTextInferInterface : public SingleReqInferInterfaceBase {
   public:
    explicit SingleReqTritonTextInferInterface(const std::shared_ptr<SingleLLMReqHandlerBase> &singleLLMReqHandlerBase,
                                               bool stream = false, std::string model = "", bool isReCompute = false,
                                               const std::vector<LoraParamSPtr> loraConfigs = {}) noexcept;
    bool BuildResponseJson(ResponseSPtr response, const std::vector<BestNTokens> &tokenIdList,
                           RespBodyQueue &jsonStrings, const uint64_t &timestamp = 0) override;
    void SetDMIReComputeBuilder() override;
    bool SetupInferParams(RequestSPtr tmpReq, std::string &msg) override;

   protected:
    bool ValidateAndPrepareReqToken(nlohmann::ordered_json &body, std::string &msg, uint64_t &timestamp) override;

   private:
    bool ParseRequestId(std::string &error) noexcept;
    void FillResponseBody(const uint64_t &curSeqId, nlohmann::ordered_json &respBody, const std::string &textOutput,
                          uint32_t tokenCnt) noexcept;
    Status DecodeOutputText(const uint64_t &curSeqId, std::string &textOutput, std::vector<int64_t> &tokens,
                            bool showFullText = false, const uint64_t &timestamp = 0);
    void TruncateOutputText(const ResponseSPtr &response, std::string &textOutput);
    std::string BuildTritonTextReComputeBody(const std::vector<BestNTokens> &tokens);
    void UpdateResponseBody(const uint64_t &curSeqId, nlohmann::ordered_json &jsonObj, std::string &textOutput,
                            std::string &jsonString);
};
}  // namespace mindie_llm

#endif  // ENDPOINT_TRITON_TEXT_INFER_H
