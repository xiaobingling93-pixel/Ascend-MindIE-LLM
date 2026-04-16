/**
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

#ifndef MINDIE_LLM_INFERENCE_REQUEST_H
#define MINDIE_LLM_INFERENCE_REQUEST_H

#include <memory>
#include <string>
#include <vector>

#include "data_type.h"
#include "infer_request_id.h"
#include "infer_tensor.h"
#include "status.h"
namespace mindie_llm {
class InferRequestImpl;
/// The InferRequest class is used to manage the input tensors of the inference process.
///
/// The InferRequest class provides methods to add and set input tensors, get request id,
/// and get the status of the request, and it also provides methods required by the prefill and decode separation.
class InferRequest {
   public:
    explicit InferRequest(InferRequestId requestId);

    /// Add a tensor to the inference request.
    /// This method adds a tensor to the inference request with the specified name. The tensor is stored in the request
    ///
    /// \param tensorName The name of the tensor to add.
    /// \param tensor The tensor to add.
    /// \return The status of the operation AddTensor.
    Status AddTensor(const std::string &tensorName, TensorPtr &tensor);

    /// Set a tensor to the inference request.
    ///
    /// This method sets a tensor to the inference request with the specified name. The tensor is stored in the request
    ///
    /// \param tensorName The name of the tensor to be set.
    /// \param tensor The tensor to be set.
    void SetTensor(const std::string &tensorName, TensorPtr &tensor);

    /// Get a tensor from the inference request with the specified name.
    ///
    /// This method gets a tensor from the inference request with the specified name,
    /// the tensor is stored in the request
    ///
    /// \param tensorName The name of the tensor to be acquired.
    /// \param tensor The tensor to be acquired.
    /// \return The status of the operation GetTensorByName.
    Status GetTensorByName(const std::string &tensorName, TensorPtr &tensor);

    /// Delete a tensor from the inference request with the specified name.
    ///
    /// This method deletes a tensor from the inference request with the specified name,
    /// the tensor is stored in the request.
    ///
    /// \param name The name of the tensor to be deleted.
    /// \return The status of the operation DelTensorByName.
    Status DelTensorByName(const std::string &name);

    /// Get the request id of the inference request.
    ///
    /// This method gets the request id of the inference request.
    ///
    /// \return The request id of the inference request.
    InferRequestId GetRequestId() const;

    /// Set the MaxOutputLen of the inference request.
    Status SetMaxOutputLen(uint32_t maxOutputLen);

    /// Get the MaxOutputLen of the inference request.
    uint32_t GetMaxOutputLen() const;

    std::shared_ptr<InferRequestImpl> GetRequestInner() const;

    /// Get the immutable inputs of the inference request.
    ///
    /// This method retrieves all tensors from the request and returns them as an tensor map.
    /// \return The collection of tensors in TensorMap format.
    const TensorMap &ImmutableInputs() const;

    /// Set the request type of the inference request.
    void SetReqType(mindie_llm::InferReqType reqType);

    /// Get the request type of the inference request.
    mindie_llm::InferReqType GetReqType() const;

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

    void SetRecompute(bool isRecompute);

    bool IsRecompute() const;

    ~InferRequest();

   private:
    std::shared_ptr<InferRequestImpl> impl_;
};
}  // namespace mindie_llm

#endif  // MINDIE_LLM_INFERENCE_REQUEST_H
