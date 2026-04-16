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

#ifndef MINDIE_LLM_CALLBACK_H
#define MINDIE_LLM_CALLBACK_H
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "infer_request.h"
#include "infer_tensor.h"
#include "status.h"
namespace mindie_llm {
/// The eum class of callback type, which includes control signal status and request enqueue status.
/// The CONTROL_SIGNAL_STATUS is used to identify that the Staus Response is for the request
/// with control operation enque status. The REQUEST_ENQUEUE_STATUS is used to identify that
/// the Staus Response is for the request enqueue status.
enum class StatusResponseType {
    CONTROL_SIGNAL_STATUS = 0,
    REQUEST_ENQUEUE_STATUS = 1,
};

/// The eum class of operation type, which includes STOP and RELEASE_KV.
/// The STOP is used to identify that the operation is to stop the request inference.
/// The RELEASE_KV is used to identify that the operation is to release the kv cache.
enum class Operation {
    STOP = 1,
    RELEASE_KV = 2,
};

/// Use the std::function to define a callback function for retriving requests.
/// The callback function should return a vector of shared_ptr of InferRequest.
///
/// \return std::vector<std::shared_ptr<InferRequest>>
using GetRequestsCallback = std::function<std::vector<std::shared_ptr<InferRequest>>()>;

/// Use the std::function to define a callback function for sending responses.
/// The callback function has 4 parameters:
///
/// \param InferRequestId request_id: The id of the request.
/// \param TensorMap TensorMap& results: The results of the request, which contains the response tensormap
/// \param bool Whether the request's inference is final.
/// \param string error_msg: The error message if the request's inference is not correctly inferenced.
using SendResponsesCallback = std::function<void(InferRequestId, const TensorMap&, bool, const std::string&)>;

/// Use the std::function to define a callback function for sending the status of requests being queued.
/// The callback function has 2 parameters:
///
/// \param InferRequestId request_id: The id of the request, whose type is InferRequestId class.
/// \param Status status: The status of requests being queued.
/// \param StatusResponseType status_type: The type of the status.
using SendStatusResponseCallback = std::function<void(InferRequestId, Status, StatusResponseType)>;

/// Use the std::function to define a callback function for retrieving the rqeuest with the given operation,
/// the operation is defined in the 'Operation' .
///
/// \returns std::vector<std::pair<InferRequestId, Operation>>: The vector of the request id and operation pair.
using ControlSignalCallback = std::function<std::vector<std::pair<InferRequestId, Operation>>()>;

/// Use the std::function to define a callback function for Getting the LLM Manager statistics.
/// The callback function has 1 parameter:
///
/// \param std::string stats: The statistics of the LLM Manager.
using LlmManagerStatsCallback = std::function<void(const std::string&)>;
}  // namespace mindie_llm
#endif  // MINDIE_LLM_CALLBACK_H
