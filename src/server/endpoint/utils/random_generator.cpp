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

#include "random_generator.h"

#include "endpoint_def.h"
#include "log.h"

namespace mindie_llm {
std::shared_ptr<RandomGenerator> RandomGenerator::gRandomGenerator = nullptr;
std::shared_mutex RandomGenerator::gInitMutex;

RandomGenerator::RandomGenerator() {
    // 使用当前时间作为种子，确保每次运行结果不同
    uint32_t seed = static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count());
    generator_.seed(seed);
}

std::shared_ptr<RandomGenerator> RandomGenerator::GetInstance() {
    {
        std::shared_lock<std::shared_mutex> readLock(gInitMutex);
        if (gRandomGenerator != nullptr) {
            return gRandomGenerator;
        }
    }
    try {
        std::unique_lock<std::shared_mutex> writeLock(gInitMutex);
        gRandomGenerator = std::make_shared<RandomGenerator>();
        return gRandomGenerator;
    } catch (std::bad_alloc& exception) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                   "Failed to new RandomGenerator, probably out of memory");
    }
    return nullptr;
}

uint32_t RandomGenerator::GetRand() noexcept { return distribution_(generator_); }

}  // namespace mindie_llm
