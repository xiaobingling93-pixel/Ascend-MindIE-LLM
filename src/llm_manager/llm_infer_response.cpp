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
#include "llm_manager/llm_infer_response.h"

#include <mutex>
using namespace mindie_llm;
namespace mindie_llm {
const mindie_llm::InferRequestId &LlmInferResponse::GetRequestId() const { return requestId_; }

void LlmInferResponse::SetEOS(bool isEos) { isEos_ = isEos; }

void LlmInferResponse::SetFlags(uint32_t flags) { flags_ = flags; }

void LlmInferResponse::SetSendResponseCallback(const mindie_llm::SendResponseCallback4Request &callback) {
    callback_ = callback;
}

const mindie_llm::SendResponseCallback4Request &LlmInferResponse::GetSendResponseCallback() const { return callback_; }

bool LlmInferResponse::IsEnd() const { return isEos_; }

uint32_t LlmInferResponse::GetFlags() const { return flags_; }

Status LlmInferResponse::GetOutput(const std::string &name, std::shared_ptr<mindie_llm::InferTensor> &tensor) const {
    std::shared_lock lock{mutex_};
    auto iter = outputs_.find(name);
    if (iter == outputs_.end()) {
        return Status(Error::Code::NOT_FOUND, "output '" + name + "' not found in response");
    }
    tensor = iter->second;
    if (tensor == nullptr) {
        return Status(Error::Code::INVALID_ARG, "tensor is nullptr in 'GetOutput' parameter");
    }
    return Status(Error::Code::OK, "Success");
}

Status LlmInferResponse::AddOutput(const std::shared_ptr<mindie_llm::InferTensor> &tensor) {
    std::unique_lock lock{mutex_};
    const auto &pr = outputs_.insert(std::make_pair(tensor->GetName(), tensor));
    if (!pr.second) {
        return Status(Error::Code::INVALID_ARG, "output '" + tensor->GetName() + "' already exists in response");
    }
    return Status(Error::Code::OK, "Success");
}

Status LlmInferResponse::DelOutput(const std::string &name) {
    if (outputs_.erase(name) != 1) {
        return Status(Error::Code::INVALID_ARG, "output '" + name + "' does not exist in response");
    }
    return Status(Error::Code::OK, "Success");
}

Status LlmInferResponse::AddMetrics(const std::vector<uint64_t> &metrics) noexcept {
    std::vector<int64_t> lshape(1, static_cast<int64_t>(metrics.size()));
    std::shared_ptr<mindie_llm::InferTensor> metricsTensor =
        std::make_shared<mindie_llm::InferTensor>("METRICS", mindie_llm::InferDataType::TYPE_UINT64, lshape);
    size_t metricsSize = sizeof(int64_t) * metrics.size();
    if (!metricsTensor->Allocate(metricsSize)) {
        return Status(Error::Code::ERROR, "failed to allocate memory for metrics tensor");
    }
    metricsTensor->SetRelease(false);
    auto *buffer = reinterpret_cast<int64_t *>(metricsTensor->GetData());
    for (std::size_t i = 0; i < metrics.size(); i++) {
        buffer[i] = metrics[i];
    }
    auto status = AddOutput(metricsTensor);
    if (!status.IsOk()) {
        return status;
    }
    return Status(Error::Code::OK, "Success");
}

uint32_t LlmInferResponse::GetIterTimes() const { return iterTimes_; }

void LlmInferResponse::SetIterTimes(uint32_t iterTimes) { iterTimes_ = iterTimes; }

uint32_t LlmInferResponse::GetTransferFlag() const { return transferFlag_; }

void LlmInferResponse::SetTransferFlag(uint32_t flag) { transferFlag_ = flag; }
}  // namespace mindie_llm
