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
#include "SensitiveInfoManager.h"

#include "log.h"
#include "memory_utils.h"

namespace mindie_llm {
SensitiveInfoManager::~SensitiveInfoManager() {
    if (content_ != nullptr) {
        try {
            Clear();
        } catch (const std::exception& e) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                       "Exception in destructor ~SensitiveInfoManager() Clear(): " << e.what());
        } catch (...) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                       "Unknown exception in destructor ~SensitiveInfoManager() Clear()");
        }
    }
}

void SensitiveInfoManager::Clear() {
    if (content_ != nullptr) {
        auto ret = memset_s(content_, len_, '\0', len_);
        if (ret != 0) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                       "Failed to clear sensitive info");
        }
        delete[] content_;
        content_ = nullptr;
        len_ = 0;
        maxLen_ = 0;
        minLen_ = 0;
    }
}

bool SensitiveInfoManager::IsValid() const { return content_ != nullptr && len_ > 0; }

// 需要再外部进行深拷贝
const char* SensitiveInfoManager::GetSensitiveInfoContent() {
    if (IsValid()) {
        return content_;
    } else {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Sensitive info is invalid");
        return nullptr;
    }
}

// 深拷贝
bool SensitiveInfoManager::CopySensitiveInfo(const char* content, size_t len) {
    if (content == nullptr) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Content is invalid");
        return false;
    }

    if (len > maxLen_ || len < minLen_) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Len is not in [" << minLen_ << ", " << maxLen_ << "]");
        return false;
    }
    // 清理内存
    Clear();
    content_ = new (std::nothrow) char[len];
    if (content_ == nullptr) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Failed to create char[]");
        return false;
    }

    len_ = len;
    auto ret = memcpy_s(content_, len_, content, len);
    if (ret != 0) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Failed to copy sensitive info");
        Clear();
        return false;
    }
    return true;
}
}  // namespace mindie_llm
