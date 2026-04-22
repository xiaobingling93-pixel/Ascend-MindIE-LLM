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

#include "thread_group_cc.h"

#include <algorithm>

using namespace std;
using namespace mindie_llm;

ThreadGroupCC &ThreadGroupCC::GetInstance(size_t numThreads) {
    if (numThreads <= 1) {
        throw std::runtime_error("the numThreads must be greated than 1.");
    }

    static ThreadGroupCC instance(numThreads);
    return instance;
}

ThreadGroupCC::ThreadGroupCC(size_t numThreads)
    : numThreads_(numThreads),
      broadcastReadyVec_(numThreads, false),
      gatherReadyVec_(numThreads, false),
      gatherReaderDoneVec_(numThreads, false),
      scatterReadyVec_(numThreads, false),
      allGatherReadyVec_(numThreads, std::vector<bool>(numThreads, false)),
      allGatherReaderDoneVec_(numThreads, std::vector<bool>(numThreads, false)),
      reduceReadyVec_(numThreads, false),
      reduceReaderDoneVec_(numThreads, false) {
    gatherBuf_.resize(numThreads);
    scatterBuf_.resize(numThreads);
    allGatherBuf_.resize(numThreads);
    reduceBuf_.resize(numThreads);
}

// 显式实例化常用类型
template void ThreadGroupCC::AllGather<int64_t>(const std::vector<int64_t> &, std::vector<std::vector<int64_t>> &,
                                                size_t);
