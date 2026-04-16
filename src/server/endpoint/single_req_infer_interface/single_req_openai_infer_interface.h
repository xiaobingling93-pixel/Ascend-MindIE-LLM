/*
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

#ifndef ENDPOINT_OPENAI_INFER_H
#define ENDPOINT_OPENAI_INFER_H

#include <vector>

#include "httplib.h"
#include "infer_param.h"
#include "single_req_vllm_openai_infer_interface.h"
using OrderedJson = nlohmann::ordered_json;

namespace mindie_llm {
/**
 * @brief OpenAi 格式的推理请求处理类
 */
class SingleReqOpenAiInferInterface : public SingleReqVllmOpenAiInferInterface {
   public:
    explicit SingleReqOpenAiInferInterface(const std::shared_ptr<SingleLLMReqHandlerBase> &singleLLMReqHandlerBase,
                                           bool isReCompute = false,
                                           const std::vector<LoraParamSPtr> loraConfigs = {}) noexcept;
    bool SetupInferParams(RequestSPtr tmpReq, std::string &msg) override;

   protected:
    bool CheckModelName(const std::string &modelName) const;
    // Resolve model name with LoRA fallback (vLLM behavior)
    bool ParseModelName(nlohmann::ordered_json &body, std::string &outModel, std::string &err) override;
};
}  // namespace mindie_llm
#endif  // ENDPOINT_OPENAI_INFER_H
