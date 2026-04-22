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

#ifndef ENDPOINT_VLLM_INFER_H
#define ENDPOINT_VLLM_INFER_H

#include <cstdint>
#include <vector>

#include "httplib.h"
#include "infer_param.h"
#include "single_req_infer_interface_base.h"

namespace mindie_llm {
/**
 * @brief Triton text 格式的推理请求处理类
 */
class SingleReqVllmInferInterface : public SingleReqInferInterfaceBase {
   public:
    explicit SingleReqVllmInferInterface(const std::shared_ptr<SingleLLMReqHandlerBase> &singleLLMReqHandlerBase,
                                         bool isReCompute = false,
                                         const std::vector<LoraParamSPtr> loraConfigs = {}) noexcept;
    bool BuildResponseJson(ResponseSPtr response, const std::vector<BestNTokens> &tempTokens,
                           RespBodyQueue &jsonStrings, const uint64_t &timestamp = 0) override;
    void SetDMIReComputeBuilder() override;
    const InferParam::FeatureSupport &GetFeatureSupport() const override {
        static constexpr InferParam::FeatureSupport kSupport{false, false, true, false};
        return kSupport;
    }
    bool SetupInferParams(RequestSPtr tmpReq, std::string &msg) override;

   protected:
    bool ValidateAndPrepareReqToken(nlohmann::ordered_json &body, std::string &msg, uint64_t &timestamp) override;
    void SendStreamResponse(RespBodyQueue &jsonStrings) override;

   private:
    bool SetReturnSeqCount(RequestSPtr req, std::string &errMsg);
    std::string ChangeUtf8Str(std::string &input) const;
    bool EncodeVllmResponse(RespBodyQueue &jsonStrs);
    bool EncodeVllmStreamResponse(RespBodyQueue &jsonStrings) noexcept;
    std::string BuildVllmReComputeBody(const std::vector<BestNTokens> &tokens);
};
}  // namespace mindie_llm

#endif  // ENDPOINT_VLLM_INFER_H
