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

#ifndef MINDIE_LLM_INFERENCE_REQUEST_IMPL_H
#define MINDIE_LLM_INFERENCE_REQUEST_IMPL_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "status.h"
#include "infer_tensor.h"
#include "infer_callback.h"
#include "callback.h"
namespace mindie_llm {
class InferRequestImpl {
public:
    explicit InferRequestImpl(InferRequestId requestId);

    Status AddTensor(const std::string& tensorName, TensorPtr &tensor);

    void SetTensor(const std::string &tensorName, TensorPtr &tensor);

    Status GetTensorByName(const std::string& tensorName, TensorPtr &tensor);

    Status DelTensorByName(const std::string &name);

    InferRequestId GetRequestId() const;

    Status SetMaxOutputLen(uint32_t maxOutputLen);

    uint32_t GetMaxOutputLen() const;

    void SetSendResponseCallback(const mindie_llm::SendResponseCallback4Request &callback);

    void SetReleaseCallback(const mindie_llm::ReleaseCallback &callback);

    mindie_llm::ReleaseCallback &GetReleaseCallback();

    mindie_llm::SendResponseCallback4Request &GetSendResponseCallback();

    void SetEngineResponseCallback(const mindie_llm::SendResponseCallback4Request &callback);

    mindie_llm::SendResponseCallback4Request &GetEngineResponseCallback();

    const TensorMap &ImmutableInputs() const;

    void SetReqType(mindie_llm::InferReqType reqType);

    mindie_llm::InferReqType GetReqType() const;

    void SetRecompute(bool isRecompute);

    bool IsRecompute() const;

    bool IsPrefillReq() const;

    bool IsDecodeReq() const;

    void SetDTarget(std::string &dTarget);

    std::string GetDTarget() const;

    void SetPrefillAddr(std::string &prefillAddr);

    std::string GetPrefillAddr() const;

    void SetSrcBlockTable(const std::vector<int64_t> &srcBlockTable);

    std::vector<int64_t> GetSrcBlockTable() const;

    void SetDpInstanceIds(const std::vector<uint64_t> &dpInstanceIds);

    std::vector<uint64_t> GetDpInstanceIds() const;

    void SetSrcHmoTable(const std::vector<std::vector<int64_t>> &srcHmoTable);

    std::vector<std::vector<int64_t>> GetSrcHmoTable() const;

private:
    InferRequestId requestId_;

    uint64_t maxOutputLen_ = 1024;

    bool hasSampling_ = false;
    mindie_llm::SendResponseCallback4Request responseCallback_{};
    mindie_llm::ReleaseCallback releaseCallback_{};
    mindie_llm::SendResponseCallback4Request engineResponseCallback_{};
    TensorMap inputs_{};

    mindie_llm::InferReqType reqType_{ mindie_llm::InferReqType::REQ_STAND_INFER };
    bool isRecompute_ = false;
    std::string dTarget_;
    std::string prefillAddr_{0};
    std::vector<std::vector<int64_t>> srcBlockTable_{}; // per block manager
    std::vector<uint64_t> dpInstanceIds_{};
    std::vector<std::vector<int64_t>> srcHmoTable_{};
};
} // namespace mindie_llm

#endif