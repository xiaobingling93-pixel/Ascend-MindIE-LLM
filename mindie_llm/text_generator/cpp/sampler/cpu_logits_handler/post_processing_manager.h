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

#ifndef MINDIE_LLM_PROCESSING_MANAGER_H
#define MINDIE_LLM_PROCESSING_MANAGER_H

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_base.h"
#include "post_processing.h"
#include "post_processing_profiler/profiler.h"
#include "thread_pool.h"

namespace py = pybind11;
#pragma GCC visibility push(default)

class PostProcessingManager {
   public:
    static std::shared_ptr<PostProcessingManager> &Instance(int threadNumIn, int deviceIdIn) {
        static std::mutex mt;
        static std::shared_ptr<PostProcessingManager> cacheInstance = nullptr;
        if (cacheInstance == nullptr) {
            std::lock_guard<std::mutex> guard(mt);
            if (cacheInstance == nullptr) {
                cacheInstance = std::make_shared<PostProcessingManager>(threadNumIn, deviceIdIn);
                cacheInstance->Init();
            }
        }
        return cacheInstance;
    }

    PostProcessingManager(int threadNumIn, int deviceIdIn) : threadNum(threadNumIn), deviceId(deviceIdIn) {};
    void Init();
    ~PostProcessingManager();

    PostProcessingManager(const PostProcessingManager &) = delete;
    PostProcessingManager &operator=(const PostProcessingManager &) = delete;

    std::pair<py::array_t<int>, py::array_t<float>> NextTokenChooser(py::array_t<int> requestIds, uint64_t scoresAddr,
                                                                     uint64_t indexAddr, int batchSize, int scoreSize,
                                                                     int maxLogprobs, std::string dtype, bool speedMode,
                                                                     bool useApproxIn);
    void SetBatchConfigs(py::array_t<int> requestIds, py::array_t<int> topK, py::array_t<float> topP,
                         py::array_t<bool> sample, py::array_t<int> numLogprobs, py::array_t<unsigned long long> seed,
                         std::string sampleMethod);
    void DeleteConf(py::array_t<int> requestIdsD);

   private:
    void UpdateRandomValue(int batchSize, int *requestIdsPtr, int requestIdsSize);

    mindie_llm::cpu_logits_handler::ThreadPool *m_pool = nullptr;
    std::map<int, mindie_llm::cpu_logits_handler::Configure> dictConf;
    py::array_t<int> result;
    py::array_t<float> logprobs;
    int threadNum;
    int deviceId;
};
#pragma GCC visibility pop
#endif
