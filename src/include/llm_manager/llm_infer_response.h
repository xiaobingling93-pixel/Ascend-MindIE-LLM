/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef MIES_LLM_INFER_RESPONSE_H
#define MIES_LLM_INFER_RESPONSE_H

#include <shared_mutex>

#include "infer_callback.h"
#include "infer_request_id.h"
#include "infer_response.h"
#include "infer_tensor.h"
#include "status.h"
namespace mindie_llm {
class LlmInferResponse : public mindie_llm::InferResponse {
   public:
    explicit LlmInferResponse(const mindie_llm::InferRequestId &reqId) noexcept : requestId_{reqId} {}

    const mindie_llm::InferRequestId &GetRequestId() const override;

    void SetEOS(bool isEos);

    void SetFlags(uint32_t flags);

    void SetSendResponseCallback(const mindie_llm::SendResponseCallback4Request &callback);

    const mindie_llm::SendResponseCallback4Request &GetSendResponseCallback() const;

    bool IsEnd() const override;

    uint32_t GetFlags() const override;

    Status GetOutput(const std::string &name, std::shared_ptr<mindie_llm::InferTensor> &tensor) const override;

    Status AddOutput(const std::shared_ptr<mindie_llm::InferTensor> &tensor);

    Status DelOutput(const std::string &name);

    Status AddMetrics(const std::vector<uint64_t> &metrics) noexcept;

    uint32_t GetIterTimes() const override;

    void SetIterTimes(uint32_t iterTimes);

    uint32_t GetTransferFlag() const override;

    void SetTransferFlag(uint32_t flag);

   private:
    std::unordered_map<std::string, std::shared_ptr<mindie_llm::InferTensor>> outputs_;
    bool isEos_{false};
    mindie_llm::InferRequestId requestId_;
    mindie_llm::SendResponseCallback4Request callback_;
    uint32_t iterTimes_{0U};

    // flags_ = 1, 请求正常结束
    // flags_ = 2, 请求被主动CANCEL或STOP，用户不感知，丢弃响应
    // flags_ = 3, 请求执行中出错，响应输出为空，err_msg非空
    // flags_ = 4, 请求输入校验异常，响应输出为空，err_msg非空
    // flags_ = 5, 请求因达到最大序列长度而结束，响应为最后一轮迭代输出
    // flags_ = 6, 请求因达到最大输出长度（包括请求和模型粒度）而结束，响应为最后一轮迭代输出
    uint32_t flags_{0U};

    // 传输状态
    uint32_t transferFlag_{0U};
    mutable std::shared_mutex mutex_;
};
}  // namespace mindie_llm
#endif  // MIES_LLM_INFER_RESPONSE_H
