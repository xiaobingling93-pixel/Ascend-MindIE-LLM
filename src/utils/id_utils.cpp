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
#include "id_utils.h"

#include <atomic>
#include <chrono>
#include <iomanip>
#include <random>
#include <sstream>

namespace mindie_llm {

std::atomic<SequenceId> g_globalSeqId{100};
std::atomic<BatchId> g_globalBatchId{0};

RequestId IDUtils::GenerateRequestID() {
    using namespace std::chrono;

    constexpr int HEX_WIDTH = 16;  // Width for 64-bit hex representation

    uint64_t nowMicro =
        static_cast<uint64_t>(duration_cast<microseconds>(system_clock::now().time_since_epoch()).count());
    uint64_t nowCpu =
        static_cast<uint64_t>(duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());

    std::seed_seq seed{static_cast<uint32_t>(nowMicro & 0xFFFFFFFF), static_cast<uint32_t>(nowMicro >> 32),
                       static_cast<uint32_t>(nowCpu & 0xFFFFFFFF), static_cast<uint32_t>(nowCpu >> 32)};

    std::mt19937_64 rng(seed);  // Initialize a 64-bit random number generator with the seed sequence

    uint64_t part1 = rng();
    uint64_t part2 = rng();

    // Concatenate the two 64-bit numbers into a 128-bit unique ID (32 hex digits)
    std::ostringstream oss;
    oss << std::hex << std::setw(HEX_WIDTH) << std::setfill('0') << part1 << std::setw(HEX_WIDTH) << std::setfill('0')
        << part2;
    return oss.str();
}

SequenceId IDUtils::GenerateSequenceId() { return g_globalSeqId.fetch_add(1); }

BatchId IDUtils::GenerateBatchID() { return g_globalBatchId.fetch_add(1); }

}  // namespace mindie_llm
