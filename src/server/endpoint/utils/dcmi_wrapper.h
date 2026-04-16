/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#pragma once
#include <dlfcn.h>

#include <mutex>
#include <string>
#include <unordered_map>

#include "file_utils.h"
#include "log.h"

namespace mindie_llm {
/**
 * @class DCMIWrapper
 * @brief DCMI（DaVinci Card Management Interface）是NPU的设备管理API接口，可以使用该接口查询NPU设备的信息和使用情况。
 *
 * 本类封装了DCMI C接口，提供以下功能：
 * 1. 动态链接libdcmi.so库并初始化dcmi接口
 * 2. 查找与加载dcmi接口函数
 * 3. 管理动态链接资源，避免内存泄露
 *
 * DCMI接口典型用途：
 * 1. 监控NPU设备AiCore、AiCPU、内存等信息
 * 2. 查询NPU设备数量、型号、算力等信息
 * 3. 详细用途链接：https://www.hiascend.com/document/detail/zh/Atlas%20200I%20A2/253RC1/re/api/api_062.html
 */
class DCMIWrapper {
   public:
    static DCMIWrapper& GetInstance();

    // 初始化/反初始化
    bool Initialize();
    void Finalize();

    // 资源清理
    void CleanUp();

    // 函数获取接口
    template <typename FuncType>
    FuncType GetFunction(const std::string& funcName) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!initialized_) {
            std::string errorMsg = "DCMI wrapper not initialized";
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_INIT, INIT_ERROR),
                       errorMsg.c_str());
            return nullptr;
        }

        return LoadFunction<FuncType>(funcName);
    }

    bool IsInitialized() const { return initialized_; }

   private:
    DCMIWrapper();
    ~DCMIWrapper();

    DCMIWrapper(const DCMIWrapper&) = delete;
    DCMIWrapper& operator=(const DCMIWrapper&) = delete;

    // 函数加载器
    template <typename FuncType>
    FuncType LoadFunction(const std::string& funcName) {
        if (!handle_) {
            return nullptr;
        }

        auto it = funcCache_.find(funcName);
        if (it != funcCache_.end()) {
            auto result = reinterpret_cast<FuncType>(it->second);
            if (!result) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_INIT, INIT_ERROR),
                           "reinterpret_cast produced null from dlsym result");
                funcCache_.erase(it);
                return nullptr;
            }
            return result;
        }

        void* funcPtr = dlsym(handle_, funcName.c_str());
        if (!funcPtr) {
            std::string errorMsg = std::string("DCMI function ") + funcName + " not found: " + dlerror();
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_INIT, INIT_ERROR),
                       errorMsg.c_str());
            return nullptr;
        }
        funcCache_[funcName] = funcPtr;
        auto result = reinterpret_cast<FuncType>(funcPtr);
        if (!result) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_INIT, INIT_ERROR),
                       "reinterpret_cast produced null from dlsym result");
            return nullptr;
        }
        return result;
    }

    // 链接库检查函数
    bool LoadAndCheckLibrary();

    void* handle_;
    bool initialized_;
    mutable std::mutex mutex_;
    std::string dcmiPath_ = "";

    // 函数指针缓存
    std::unordered_map<std::string, void*> funcCache_;
};
}  // namespace mindie_llm
