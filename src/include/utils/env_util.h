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

#ifndef MIES_ENV_UTIL_H
#define MIES_ENV_UTIL_H

#include <cstdint>
#include <map>
#include <string>

namespace mindie_llm {
class EnvUtil {
   public:
    static EnvUtil& GetInstance() {
        static EnvUtil instance;
        return instance;
    }
    const std::string& Get(const std::string& name) const;
    std::string GetEnvByName(const std::string& name) const;

    /**
     * @brief 获取指定环境变量并解析为 int32_t
     * @param name 环境变量名称
     * @param defaultValue 解析失败或环境变量不存在时返回的默认值
     * @return 解析成功返回对应整数值；解析失败或不存在时返回 defaultValue
     */
    int32_t GetInt(const std::string& name, int32_t defaultValue = 0) const;
    void SetEnvVar(const std::string& name, const std::string& value);
    void ClearEnvVar(const std::string& name);

    EnvUtil(const EnvUtil&) = delete;
    EnvUtil& operator=(const EnvUtil&) = delete;
    ~EnvUtil() = default;

   private:
    EnvUtil();
    std::map<std::string, std::string> vars;
};
}  // namespace mindie_llm

#endif  // MIES_ENV_UTIL_H
