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

#ifndef MINDIE_DYNAMIC_CONFIG_HANDLER_H
#define MINDIE_DYNAMIC_CONFIG_HANDLER_H

#include <atomic>
#include <functional>
#include <mutex>
#include <nlohmann/json.hpp>
#include <vector>

namespace mindie_llm {

class DynamicConfigHandler {
   public:
    static DynamicConfigHandler &GetInstance();
    void Start() const;
    void Stop() const;

    template <typename T>
    void RegisterCallBackFunction(const std::string &pathExpression, T *obj, void (T::*method)(uint64_t),
                                  uint64_t value) const {
        std::lock_guard<std::mutex> locker(GetInstance().vectorMutex);
        GetInstance().callBackFunctions.push_back(
            std::make_pair(pathExpression, [obj, method, value] { (obj->*method)(value); }));
    }

   private:
    DynamicConfigHandler() {}
    ~DynamicConfigHandler();
    std::vector<std::string> splitString(const std::string &s, const char delimiter = '.') const;
    std::string getConfigFilePath() const;
    bool CheckSystemConfig(const std::string &jsonPath, nlohmann::json &inputJsonData, std::string paramType) const;
    bool isTriggered(const std::string pathExpression) const;

    std::vector<std::pair<std::string, std::function<void()>>> callBackFunctions;
    bool isRunning = true;
    std::mutex vectorMutex;
};

}  // namespace mindie_llm

#endif  // MINDIE_DYNAMIC_CONFIG_HANDLER_H
