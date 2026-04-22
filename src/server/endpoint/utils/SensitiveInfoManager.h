/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef SENSITIVE_INFO_MANAGER_H
#define SENSITIVE_INFO_MANAGER_H
#include <unistd.h>

#include <cstdint>

namespace mindie_llm {
class SensitiveInfoManager {
   public:
    SensitiveInfoManager(char* content, uint32_t len, uint32_t maxLen, uint32_t minLen)
        : content_(content), len_(len), maxLen_(maxLen), minLen_(minLen) {}
    ~SensitiveInfoManager();

    void Clear();
    bool IsValid() const;
    // 需要再外部进行深拷贝
    const char* GetSensitiveInfoContent();
    // 深拷贝
    bool CopySensitiveInfo(const char* content, size_t len);

   private:
    char* content_{nullptr};
    uint32_t len_{0};
    uint32_t maxLen_{0};
    uint32_t minLen_{0};
};
}  // namespace mindie_llm

#endif  // SENSITIVE_INFO_MANAGER_H
