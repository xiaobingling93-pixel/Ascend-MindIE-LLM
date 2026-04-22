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

#include "dcmi_wrapper.h"

#include "dcmi_interface_api.h"
#include "log.h"

namespace mindie_llm {
DCMIWrapper::DCMIWrapper() : handle_(nullptr), initialized_(false) {}

DCMIWrapper::~DCMIWrapper() { Finalize(); }

DCMIWrapper& DCMIWrapper::GetInstance() {
    static DCMIWrapper instance;
    return instance;
}

bool DCMIWrapper::Initialize() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) {
        return true;
    }

    // 加载并检查链接库libdcmi.so
    if (!LoadAndCheckLibrary()) {
        return false;
    }

    // 执行初始化函数dcmi_init
    using DcmiInitFunc = int (*)();
    DcmiInitFunc dcmiInit = LoadFunction<DcmiInitFunc>("dcmi_init");
    if (!dcmiInit) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_INIT, INIT_ERROR),
                   "dcmi_init conversion failed");
        CleanUp();
        return false;
    }
    int ret = dcmiInit();
    if (ret != 0) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_INIT, INIT_ERROR),
                   "DCMI init failed, error code: [" << ret << "]");
        CleanUp();
        return false;
    }

    initialized_ = true;
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "DCMI wrapper initialized successfully");
    return true;
}

void DCMIWrapper::Finalize() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        return;
    }

    initialized_ = false;
    CleanUp();
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "DCMI wrapper finalized");
}

bool DCMIWrapper::LoadAndCheckLibrary() {
    // 加载链接库libdcmi.so和初始化函数
    handle_ = dlopen("libdcmi.so", RTLD_LAZY);
    if (!handle_) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_INIT, INIT_ERROR),
                   "Failed to load libdcmi.so: " << dlerror());
        return false;
    }
    void* funcPtr = dlsym(handle_, "dcmi_init");
    if (!funcPtr) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_INIT, INIT_ERROR),
                   "DCMI function dcmi_init not found " << dlerror());
        CleanUp();
        return false;
    }

    // 获取链接库路径
    Dl_info info;
    if (dladdr(funcPtr, &info) == 0 || info.dli_fname == nullptr) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_INIT, INIT_ERROR),
                   "dladdr failed to get library info");
        CleanUp();
        return false;
    }

    // 校验路径合法性
    std::string errMsg;
    std::string regularPath;
    const mode_t onlyReadMode = 0b100'100'100;  // libdcmi.so文件权限为444
    if (!FileUtils::RegularFilePath(info.dli_fname, "/", errMsg, regularPath) ||
        !FileUtils::IsFileValid(regularPath, errMsg, true, onlyReadMode, false) ||
        !FileUtils::ConstrainPermission(regularPath, onlyReadMode, errMsg)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_INIT, INIT_ERROR), errMsg);
        CleanUp();
        return false;
    }

    // 缓存初始化函数
    funcCache_["dcmi_init"] = funcPtr;
    return true;
}

void DCMIWrapper::CleanUp() {
    funcCache_.clear();
    if (handle_) {
        dlclose(handle_);
        handle_ = nullptr;
    }
}
}  // namespace mindie_llm
