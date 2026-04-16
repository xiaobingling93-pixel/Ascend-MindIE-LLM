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

#ifndef GRPC_WRAPPER_H
#define GRPC_WRAPPER_H

#include <atomic>

#include "endpoint_def.h"

namespace mindie_llm {
class GrpcWrapper {
   public:
    static GrpcWrapper& GetInstance() {
        static GrpcWrapper instance;
        return instance;
    }

    int32_t Start();
    void Stop();

   private:
    GrpcWrapper() = default;
    ~GrpcWrapper() = default;
    std::atomic<bool> started_{false};
};
}  // namespace mindie_llm

#endif  // GRPC_WRAPPER_H
