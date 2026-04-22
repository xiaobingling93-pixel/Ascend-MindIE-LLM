/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef OCK_ENDPOINT_HTTP_WRAPPER_H
#define OCK_ENDPOINT_HTTP_WRAPPER_H

#include <functional>
#include <map>
#include <memory>
#include <mutex>

namespace mindie_llm {
class HttpWrapper {
   public:
    static HttpWrapper& Instance() {
        static HttpWrapper instance;
        return instance;
    }

    bool Start();
    void Stop();

   private:
    HttpWrapper() = default;
    ~HttpWrapper() = default;
    HttpWrapper(const HttpWrapper&) = delete;
    HttpWrapper& operator=(const HttpWrapper&) = delete;

    std::mutex mMutex;
    bool mStarted{false};
};
}  // namespace mindie_llm

#endif  // OCK_ENDPOINT_HTTP_WRAPPER_H
