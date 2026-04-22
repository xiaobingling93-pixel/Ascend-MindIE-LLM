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

#ifndef GRPC_HANDLER_H
#define GRPC_HANDLER_H

#include <atomic>

namespace mindie_llm {
class GrpcHandler {
   public:
    static GrpcHandler& GetInstance();
    // 注册 dmi 业务回调到 grpc service
    bool InitDmiBusiness();
    // 启动服务
    bool InitGrpcService();

   private:
    GrpcHandler() = default;
    ~GrpcHandler() = default;
    GrpcHandler(const GrpcHandler&) = delete;
    GrpcHandler& operator=(const GrpcHandler&) = delete;
    std::atomic<bool> isReady_{false};
};
}  // namespace mindie_llm

#endif  // GRPC_HANDLER_H
