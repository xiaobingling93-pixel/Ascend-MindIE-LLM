/**
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

#ifndef ENDPOINT_TGI_TEXT_INFER_H
#define ENDPOINT_TGI_TEXT_INFER_H

#include <cstdint>
#include <vector>

#include "httplib.h"
#include "infer_param.h"
#include "parse_protocol.h"
#include "single_req_infer_interface_base.h"

namespace mindie_llm {
/**
 * @brief Triton token 格式的推理请求处理类
 */
class SingleReqTgiTextInferInterface : public SingleReqInferInterfaceBase {
   public:
    explicit SingleReqTgiTextInferInterface(const std::shared_ptr<SingleLLMReqHandlerBase> &singleLLMReqHandlerBase,
                                            bool isReCompute = false, bool stream = false,
                                            const std::vector<LoraParamSPtr> loraConfigs = {}) noexcept;
    bool BuildResponseJson(ResponseSPtr response, const std::vector<BestNTokens> &tempTokens, RespBodyQueue &jsonObjs,
                           const uint64_t &timestamp = 0) override;
    bool SetupInferParams(RequestSPtr tmpReq, std::string &msg) override;
    void SetDMIReComputeBuilder() override;

   protected:
    virtual bool EncodeTGIResponse(RespBodyQueue &jsonStrs);
    bool ValidateAndPrepareReqToken(nlohmann::ordered_json &body, std::string &msg, uint64_t &timestamp) override;
    void SendStreamResponse(RespBodyQueue &jsonStrs) override;
    bool ValidTGIParameterSpec(std::string &msg);
    void TruncateReqTokens();
    int32_t GenerateRspDetailJsonStr(nlohmann::ordered_json &jsonObj, std::string &jsonStr);
    int32_t EncodeTGIStreamResponse(RespBodyQueue &jsonStrs);
    bool AssignAdapterId(const nlohmann::ordered_json &body, RequestSPtr tmpReq, std::string &error) const;
    std::string BuildTgiReComputeBody(const std::vector<BestNTokens> &tokens);
    void ParseStopString(nlohmann::ordered_json &newReqJsonObj);
    std::string ChangeUtf8Str(std::string &input) const;

    bool decoderInputDetails{false};
    uint32_t truncate = 0;
};

class SingleReqGeneralTgiTextInferInterface : public SingleReqTgiTextInferInterface {
   public:
    explicit SingleReqGeneralTgiTextInferInterface(
        const std::shared_ptr<SingleLLMReqHandlerBase> &singleLLMReqHandlerBase, bool isReCompute = false,
        bool stream = false, const std::vector<LoraParamSPtr> loraConfigs = {}) noexcept
        : SingleReqTgiTextInferInterface(singleLLMReqHandlerBase, isReCompute, stream, loraConfigs) {
        inputParam->streamMode = stream;
    }

   protected:
    bool EncodeTGIResponse(RespBodyQueue &jsonStrs) override;
};
}  // namespace mindie_llm

#endif  // ENDPOINT_TGI_TEXT_INFER_H
