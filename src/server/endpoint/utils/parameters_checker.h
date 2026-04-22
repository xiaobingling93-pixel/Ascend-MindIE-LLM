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

#ifndef PARAMETERS_CHECKER_H
#define PARAMETERS_CHECKER_H

#include <optional>
#include <string>
#include <type_traits>

#include "nlohmann/json.hpp"

// 类型特征：检查类型是否为 std::optional
template <typename T>
struct is_optional : std::false_type {};

template <typename T>
struct is_optional<std::optional<T>> : std::true_type {};

template <typename T>
constexpr bool is_optional_v = is_optional<T>::value;

namespace mindie_llm {
class ParametersChecker {
   public:
    static bool OptionalBooleanJsonCheck(const nlohmann::ordered_json &jsonObj, const std::string &key,
                                         std::optional<bool> &output, std::string &error);
    static bool BooleanJsonCheck(const nlohmann::ordered_json &jsonObj, const std::string &key, bool &output,
                                 std::string &error);
    static bool OptionalFloatJsonCheck(
        const nlohmann::ordered_json &jsonObj, const std::string &key, std::optional<float> &output, std::string &error,
        const std::function<bool(const double &value, std::stringstream &errorStream)> &validator);
    static bool FloatJsonCheck(
        const nlohmann::ordered_json &jsonObj, const std::string &key, float &output, std::string &error,
        const std::function<bool(const float &value, std::stringstream &errorStream)> &validator);
    static bool Int32JsonCheck(
        const nlohmann::ordered_json &jsonObj, const std::string &key, int32_t &output, std::string &error,
        const std::function<bool(const int64_t &value, std::stringstream &errorStream)> &validator);
    static bool OptionalInt32JsonCheck(
        const nlohmann::ordered_json &jsonObj, const std::string &key, std::optional<int32_t> &output,
        std::string &error, const std::function<bool(const int64_t &value, std::stringstream &errorStream)> &validator);
    static bool UInt64JsonCheck(
        const nlohmann::ordered_json &jsonObj, const std::string &key, uint64_t &output, std::string &error,
        const std::function<bool(const uint64_t &value, std::stringstream &errorStream)> &validator);
    static bool OptionalUInt64JsonCheck(
        const nlohmann::ordered_json &jsonObj, const std::string &key, std::optional<uint64_t> &output,
        std::string &error,
        const std::function<bool(const uint64_t &value, std::stringstream &errorStream)> &validator);
    static bool OptionalUint32JsonCheck(
        const nlohmann::ordered_json &jsonObj, const std::string &key, std::optional<uint32_t> &output,
        std::string &error,
        const std::function<bool(const uint32_t &value, std::stringstream &errorStream)> &validator);
};
}  // namespace mindie_llm

#endif  // PARAMETERS_CHECKER_H
