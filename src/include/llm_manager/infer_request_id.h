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

#ifndef MINDIE_LLM_INFERENCE_REQUEST_ID_H
#define MINDIE_LLM_INFERENCE_REQUEST_ID_H

#include <functional>

namespace mindie_llm {
/// class InferRequestId
///
/// This class is used to generate inference request id, which can be either a uint64_t or a string.
/// It also implements the custom comparison operator to compare two InferRequestId objects and hash function
/// to enable proper storage and retrieval of InferRequestId objects in hash-based containers.
class InferRequestId {
   public:
    enum class DataType { UINT64, STRING };

    /// The default constructor initializes the requestLabel_ to an empty string,
    /// requestIndex_ to 0, and idType_ to UINT64.
    explicit InferRequestId() : requestLabel_(""), requestIndex_(0), idType_(InferRequestId::DataType::UINT64) {}

    /// The constructor with a string parameter initializes the requestLabel_ to the given string,
    /// requestIndex_ to 0, and idType_ to STRING.
    ///
    /// \param requestLabel The string is used to initialize requestLabel_.
    explicit InferRequestId(std::string requestLabel)
        : requestLabel_(std::move(requestLabel)), requestIndex_(0), idType_(InferRequestId::DataType::STRING) {}

    /// The constructor with a uint64_t parameter initializes the requestLabel_ to an empty string,
    /// requestIndex_ to the given value, and idType_ to UINT64.
    ///
    /// \param requestIndex The uint64_t value to used to initialize requestIndex_.
    explicit InferRequestId(uint64_t requestIndex)
        : requestIndex_(requestIndex), idType_(InferRequestId::DataType::UINT64) {}

    /// The assignment operator with a uint64_t parameter sets the requestLabel_ to an empty string,
    /// requestIndex_ to the given value, and idType_ to UINT64.
    ///
    /// \param rhs The uint64_t value is used to assign to requestIndex_.
    InferRequestId &operator=(const uint64_t rhs) {
        requestLabel_ = "";
        requestIndex_ = rhs;
        idType_ = InferRequestId::DataType::UINT64;
        return *this;
    }

    /// The assignment operator with a string parameter sets the requestLabel_ to the given value, requestIndex_ to 0,
    /// and idType_ to STRING.
    ///
    /// \param rhs The string value is used to assign to requestLabel_.
    InferRequestId &operator=(const std::string &rhs) {
        requestLabel_ = rhs;
        requestIndex_ = 0;
        idType_ = InferRequestId::DataType::STRING;
        return *this;
    }

    /// The assignment operator with a const InferRequestId parameter sets the requestLabel_ and
    /// requestIndex_ to the values of the given object, and idType_ to the value of the given object.
    ///
    /// \param rhs The const InferRequestId object to assign to this object.
    InferRequestId &operator=(const InferRequestId &rhs) {
        if (this != &rhs) {
            requestLabel_ = rhs.requestLabel_;
            requestIndex_ = rhs.requestIndex_;
            idType_ = rhs.idType_;
        }
        return *this;
    }

    /// The copy constructor creates a new InferRequestId object with the same values as the given object.
    ///
    /// \param other The InferRequestId object to copy.
    InferRequestId(const InferRequestId &other) {
        requestLabel_ = other.requestLabel_;
        requestIndex_ = other.requestIndex_;
        idType_ = other.idType_;
    }

    /// The function returns the type of the request ID.
    DataType Type() const { return idType_; }

    /// The function returns the string value of the request ID.
    const std::string &StringValue() const { return requestLabel_; }

    /// The function returns the unsigned integer value of the request ID.
    uint64_t UnsignedIntValue() const { return requestIndex_; }

    /// The function returns the string representation of the request ID.
    const std::string GetRequestIdString() const {
        if (idType_ == InferRequestId::DataType::UINT64) {
            return std::to_string(requestIndex_);
        }
        return requestLabel_;
    }

    /// The struct Compare is used to compare two InferRequestId objects.
    ///
    /// \param lhs The first InferRequestId object to compare.
    /// \param rhs The second InferRequestId object to compare.
    struct Compare {
        bool operator()(const InferRequestId &lhs, const InferRequestId &rhs) const {
            if (lhs.Type() == InferRequestId::DataType::STRING) {
                return std::hash<std::string>()(lhs.StringValue()) < std::hash<std::string>()(rhs.StringValue());
            } else {
                return lhs.UnsignedIntValue() < rhs.UnsignedIntValue();
            }
        }
    };

   private:
    /// The equal operator is used to compare two InferRequestId objects.
    ///
    /// \param lhs The first InferRequestId object to compare.
    /// \param rhs The second InferRequestId object to compare.
    /// \return true if the two objects are equal, false otherwise.
    friend bool operator==(const InferRequestId lhs, const InferRequestId rhs) {
        if (lhs.Type() == rhs.Type()) {
            switch (lhs.Type()) {
                case InferRequestId::DataType::STRING:
                    return lhs.StringValue() == rhs.StringValue();
                case InferRequestId::DataType::UINT64:
                    return lhs.UnsignedIntValue() == rhs.UnsignedIntValue();
                default:
                    return lhs.UnsignedIntValue() == rhs.UnsignedIntValue();
            }
        } else {
            return false;
        }
    }

    friend bool operator!=(const InferRequestId lhs, const InferRequestId rhs) { return !(lhs == rhs); }

    std::string requestLabel_{};  /// The label of the request.

    uint64_t requestIndex_{};  /// The index of the request.

    DataType idType_;  /// The type of the request ID.
};
}  // namespace mindie_llm

/// Hash function for the InferRequestId class,
/// depending on the type of the request ID(either string or unsigned integer),
/// it will hash the string value or the unsigned integer value.
///
/// \param reqId The InferRequestId object to hash.
/// \return The hash value of the InferRequestId object.
namespace std {
template <>
struct hash<mindie_llm::InferRequestId> {
    size_t operator()(const mindie_llm::InferRequestId &reqId) const {
        if (reqId.Type() == mindie_llm::InferRequestId::DataType::STRING) {
            return std::hash<std::string>()(reqId.StringValue());
        } else {
            return std::hash<uint64_t>()(reqId.UnsignedIntValue());
        }
    }
};
}  // namespace std

#endif  // MINDIE_LLM_INFERENCE_REQUEST_ID_H
