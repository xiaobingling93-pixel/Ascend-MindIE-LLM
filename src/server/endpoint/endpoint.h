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

#ifndef OCK_ENDPOINT_H
#define OCK_ENDPOINT_H

#include "endpoint_def.h"
#include "health_checker/health_checker.h"
#include "http_rest_resource.h"

namespace mindie_llm {
class HttpWrapper;
class EngineWrapper;

class EndPoint {
   public:
    int32_t Start(std::unordered_map<std::string, std::string> args);
    HealthChecker& GetHealthcheckerInstance() const;
    void Stop();

   private:
    int StartEndpoint();
    bool StartDynamicConfigHandler() const;
    int StartHealthChecker();
    std::mutex mMutex;
    bool mExpertParallel{false};
    bool mHealthcheckerStarted{false};
    bool mEngineStarted{false};
    bool mServerStarted{false};
    bool mTokenizerStarted{false};
};
}  // namespace mindie_llm

#endif  // OCK_ENDPOINT_H
