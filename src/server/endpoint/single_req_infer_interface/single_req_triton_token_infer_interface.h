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

#ifndef ENDPOINT_TRITON_TOKEN_INFER_H
#define ENDPOINT_TRITON_TOKEN_INFER_H

#include <cstdint>
#include <vector>

#include "infer_param.h"
#include "single_req_infer_interface_base.h"

namespace mindie_llm {
struct DoubleSearchMapping {
    std::unordered_map<InferDataType, std::string> typeToString;
    std::unordered_map<std::string, InferDataType> stringToType;
    explicit DoubleSearchMapping(std::unordered_map<InferDataType, std::string> input) noexcept
        : typeToString{std::move(input)} {
        for (auto &it : typeToString) {
            stringToType.emplace(it.second, it.first);
        }
    }
};

const DoubleSearchMapping DATA_TYPE_MAPPING{
    std::unordered_map<InferDataType, std::string>{{InferDataType::TYPE_BOOL, "BOOL"},
                                                   {InferDataType::TYPE_UINT8, "UINT8"},
                                                   {InferDataType::TYPE_UINT16, "UINT16"},
                                                   {InferDataType::TYPE_UINT32, "UINT32"},
                                                   {InferDataType::TYPE_UINT64, "UINT64"},
                                                   {InferDataType::TYPE_INT8, "INT8"},
                                                   {InferDataType::TYPE_INT16, "INT16"},
                                                   {InferDataType::TYPE_INT32, "INT32"},
                                                   {InferDataType::TYPE_INT64, "INT64"},
                                                   {InferDataType::TYPE_FP16, "FP16"},
                                                   {InferDataType::TYPE_FP32, "FP32"},
                                                   {InferDataType::TYPE_FP64, "FP64"},
                                                   {InferDataType::TYPE_STRING, "STRING"},
                                                   {InferDataType::TYPE_BF16, "BF16"}}};

/**
 * @brief Triton token 格式的推理请求处理类
 */
class SingleReqTritonTokenInferInterface : public SingleReqInferInterfaceBase {
   public:
    explicit SingleReqTritonTokenInferInterface(const std::shared_ptr<SingleLLMReqHandlerBase> &singleLLMReqHandlerBase,
                                                bool isReCompute = false,
                                                const std::vector<LoraParamSPtr> loraConfigs = {}) noexcept;
    bool BuildResponseJson(ResponseSPtr response, const std::vector<BestNTokens> &tempTokens, RespBodyQueue &jsonObjs,
                           const uint64_t &timestamp = 0) override;
    bool SetupInferParams(RequestSPtr tmpReq, std::string &msg) override;

   protected:
    bool ValidateAndPrepareReqToken(nlohmann::ordered_json &body, std::string &msg, uint64_t &timestamp) override;
    void SetDMIReComputeBuilder() override;
    void BuildReComputeInput(std::vector<int64_t> &inputTokens);

   private:
    bool CheckReqInputName(nlohmann::ordered_json &body, std::string &msg);
    bool CheckReqInputShape(nlohmann::ordered_json &body, std::string &msg);
    bool CheckReqInputDataType(nlohmann::ordered_json &body, std::string &msg);
    bool CheckReqInputData(nlohmann::ordered_json &body, std::string &msg);
    bool CheckReqId(nlohmann::ordered_json &body, std::string &msg);
    bool CheckOutputs(nlohmann::ordered_json &body, std::string &msg);
    bool CheckTritonParameter(nlohmann::ordered_json &body, std::string &msg);
    void BuildReComputeBodySampling(const uint64_t &curSeqId, OrderedJson &parameters);
    std::string BuildTritonTokenReComputeBody(const std::vector<BestNTokens> &tokens);

   private:
    std::vector<std::string> inputNames;
    std::vector<std::vector<int64_t>> inputShape;
    std::vector<InferDataType> inputDataType;
    std::vector<std::string> outputNames;
    static const std::string defOutputName;
};
}  // namespace mindie_llm

#endif  // ENDPOINT_TRITON_TOKEN_INFER_H
