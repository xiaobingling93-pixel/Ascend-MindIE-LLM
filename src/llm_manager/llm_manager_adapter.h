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

#ifndef LLM_MANAGER_ADAPTER_H_
#define LLM_MANAGER_ADAPTER_H_
#include "llm_manager/llm_manager.h"
#include "llm_manager_v2/llm_manager_v2.h"
#include "src/llm_manager_v2/include/impl/llm_manager_impl.h"

namespace mindie_llm {
std::vector<std::shared_ptr<Request>> AdaptGetRequestV1ToV2(GetRequestsCallback getRequest);

void AdaptSendResponseV2ToV1(SendResponsesCallback sendResponse, std::shared_ptr<Response> response);

std::vector<std::pair<RequestIdNew, OperationV2>> AdaptControlSignalCallbackV1ToV2(
    ControlSignalCallback controlCallback);

void AdaptStatusResponseCallbackV2ToV1(SendStatusResponseCallback statusResponseCallback, RequestIdNew requestId,
                                       Status status, StatusResponseTypeV2 statusType);

}  // namespace mindie_llm
#endif  // LLM_MANAGER_ADAPTER_H_
