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

#ifndef MIES_RANDOM_GENERATOR_H
#define MIES_RANDOM_GENERATOR_H

#include <algorithm>
#include <boost/algorithm/hex.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <shared_mutex>
namespace mindie_llm {
class RandomGenerator {
   public:
    RandomGenerator();
    ~RandomGenerator() = default;

   public:
    uint32_t GetRand() noexcept;
    static std::shared_ptr<RandomGenerator> GetInstance();

   private:
    std::mt19937 generator_;
    std::uniform_int_distribution<uint32_t> distribution_;
    static std::shared_mutex gInitMutex;
    static std::shared_ptr<RandomGenerator> gRandomGenerator;
};

inline std::string GenerateHTTPRequestUUID() {
    boost::uuids::uuid reqId = boost::uuids::random_generator()();
    std::string reqIdStr = boost::uuids::to_string(reqId);
    reqIdStr.erase(std::remove(reqIdStr.begin(), reqIdStr.end(), '-'), reqIdStr.end());

    return reqIdStr;
}
}  // namespace mindie_llm

#endif  // MIES_RANDOM_GENERATOR_H
