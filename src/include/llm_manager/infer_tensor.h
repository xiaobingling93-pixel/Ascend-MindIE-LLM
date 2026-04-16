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

#ifndef MINDIE_LLM_INFER_TENSOR_H
#define MINDIE_LLM_INFER_TENSOR_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "data_type.h"

namespace mindie_llm {
/// The class called InferTensor is used to construct an n-dimensional array of values.
///
/// This class is designed to create basic data, provides basic functions such as create/delete tensor,
/// get shape, get size, get date type, truncate,etc
class InferTensor {
   public:
    /// The default constructor of InferTensor. It initializes the tensor with default values.
    InferTensor() = default;

    /// Create a Tensor of the given `name`, `dataType` and `dataShape`
    ///
    /// \param name  The name of the tensor
    /// \param dataType The data type of the tensor
    /// \param dataShape The shape of the tensor
    InferTensor(std::string name, InferDataType dataType, std::vector<int64_t> dataShape);
    InferTensor(const InferTensor &other) = delete;
    InferTensor &operator=(const InferTensor &other) = delete;
    /// Get Shape of the Tensor
    ///
    /// \return the shape of the tensor, whose type is std::vector<int64_t>
    const std::vector<int64_t> &GetShape() const;

    /// Get Size of the Tensor
    size_t GetSize() const;

    /// Get Data Type of the Tensor
    InferDataType GetDataType() const;

    /// Get Memory Type of the Tensor
    MemType GetMemType() const;

    /// This function is used to get the data of the tensor, return a void pointer.
    void *GetData() const;

    /// This function is used to get the name of the tensor.
    const std::string &GetName() const;

    /// Truncate the Tensor to the given `truncLen` size
    ///
    /// This function provides a way to truncate the tensor to a smaller size,
    /// which is useful when the tensor is too large
    ///
    /// \param truncLen The length to truncate to
    /// \return true if the truncation is successful, false otherwise
    bool Truncate(const size_t truncLen);

    /// Allocate memory for the tensor
    ///
    /// This function allocates memory for the tensor, which is necessary before the tensor can be used.
    ///
    /// \param size The size of the memory to allocate
    ///
    /// \return true if the allocation is successful, false otherwise
    bool Allocate(size_t size);

    /// Set the buffer of the tensor
    ///
    /// This function sets the buffer of the tensor,
    /// which is useful when the tensor is already allocated and needs to be used.
    ///
    /// \param buffer The buffer to set
    /// \param tensorbyteSize The size of the buffer
    /// \param tensorNeedRelease Whether the buffer needs to be released after use
    ///
    /// \return true if the buffer is set successfully, false otherwise
    void SetBuffer(const void *buffer, size_t tensorbyteSize, bool tensorNeedRelease);

    /// This function is used to set the release flag of the tensor.
    void SetRelease(bool releaseFlag);

    void Release();

    /// This function is used to get the byte size of the dataType of the tensor.
    static size_t GetTypeByteSize(InferDataType inferDataType);

    ~InferTensor();

   private:
    /// The name of the tensor, whose private member variable stores the name of the tensor
    std::string name;
    /// The data type of the tensor, this private member variable stores the data type of the tensor
    InferDataType dataType = InferDataType::TYPE_INVALID;
    /// The shape of the tensor,  this private member variable stores the shape of the tensor
    std::vector<int64_t> dataShape;

    /// The data of the tensor, this private member variable stores the buffer of the tensor
    void *data = nullptr;
    /// The size of the data of the tensor, this private member variable stores the size of the buffer of the tensor
    uint64_t byteSize = 0;
    /// Whether the buffer needs to be released after use
    bool needRelease = false;
};

using TensorPtr = std::shared_ptr<InferTensor>;
using TensorMap = std::unordered_map<std::string, std::shared_ptr<InferTensor>>;
}  // namespace mindie_llm
#endif  // MINDIE_LLM_INFER_TENSOR_H
