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

#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H

#include <fcntl.h>
#include <semaphore.h>
#include <unistd.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "securec.h"
namespace mindie_llm {
const int MASK = 0600;

const uint32_t PROCESS = 0;
const uint32_t SHM_TOTAL_NUM = 3U;
const uint32_t SHM_BROADCAST_IDX = 0;
const uint32_t SHM_RECEIVE_IDX = 1;
const uint32_t SHM_TRANSFER_IDX = 2;

const uint32_t FIRSTCHANNEL = 0;
const uint32_t SECONDCHANNEL = 1;

constexpr int UNHEALTHY_THRESHOLD = 3;           // 3次没有收到心跳判定为异常
constexpr uint32_t MAX_INIT_RESULTS_SIZE = 200;  // 初始化模型initresult最大size

struct SharedSemaphore {
    sem_t semProduce;
    sem_t semConsume;
};

void SemP(sem_t &sem, short int num);

void SemV(sem_t &sem, short int num);

bool CheckMemorySize(const char *pointer, uint64_t byteSize, const char *start, const char *end);

int MemcpyAfterCheck(char *dest, uint64_t destBufLen, uint64_t destCurPos, const char *src, size_t count);

// IsValidNonNegativeInteger checks the worker initialization result from the slave workers.
bool IsValidNonNegativeInteger(const std::string &input);

using PVOID = void *;
using CPVOID = const void *;
using PCHAR = char *;
using CPCHAR = const char *;

#ifndef errno_t
using errno_t = int;
#endif

#ifndef EOK
constexpr int EOK = 0;
#endif
}  // namespace mindie_llm
#endif
