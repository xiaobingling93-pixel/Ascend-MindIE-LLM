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

#ifndef LLM_MANAGER_UTILS_INFER_CALLBACK_H
#define LLM_MANAGER_UTILS_INFER_CALLBACK_H
#include "infer_request_id.h"
#include "infer_response.h"

namespace mindie_llm {

using SendResponseCallback4Request = std::function<void(std::shared_ptr<InferResponse> &)>;

using ReleaseCallback = std::function<void(const InferRequestId &)>;

}  // namespace mindie_llm
#endif
