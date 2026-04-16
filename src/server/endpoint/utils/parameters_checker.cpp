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

#include "parameters_checker.h"

#include "parse_protocol.h"

using OrderedJson = nlohmann::ordered_json;

namespace mindie_llm {

namespace {
template <typename ValueType, OrderedJson::value_t JsonType>
bool GenericOptionalJsonCheck(const OrderedJson &jsonObj, const std::string &key, std::optional<ValueType> &value,
                              std::string &error) {
    auto res = JsonParse::CheckOptionalItemType(jsonObj, key, JsonType, error);
    if (!res.isCorrectType) {
        return false;
    }
    if (res.isPresent) {
        value = jsonObj[key];
    }
    return true;
}

template <typename ValueType, OrderedJson::value_t JsonType>
bool GenericRequiredJsonCheck(const OrderedJson &jsonObj, const std::string &key, ValueType &value,
                              std::string &error) {
    auto res = JsonParse::CheckOptionalItemType(jsonObj, key, JsonType, error);
    if (!res.isCorrectType) {
        return false;
    }
    if (res.isPresent) {
        value = jsonObj[key];
    }
    return true;
}

template <typename OutputType, typename ValueType, OrderedJson::value_t JsonType>
bool NullableJsonCheck(const OrderedJson &jsonObj, const std::string &key, std::string &error, OutputType &output,
                       const std::function<bool(const ValueType &value, std::stringstream &errorStream)> &validator) {
    if (!validator) {
        return true;
    }
    auto res = JsonParse::CheckOptionalItemType(jsonObj, key, JsonType, error);
    if (!res.isCorrectType) {
        return false;
    }
    if (!res.isPresent) {
        return true;
    }

    ValueType value = jsonObj[key];

    std::stringstream stream;
    if (!validator(value, stream)) {
        error = stream.str();
        return false;
    }

    if constexpr (is_optional_v<OutputType>) {
        output = static_cast<typename OutputType::value_type>(value);
    } else {
        output = static_cast<OutputType>(value);
    }
    return true;
}
}  // namespace

bool ParametersChecker::OptionalBooleanJsonCheck(const OrderedJson &jsonObj, const std::string &key,
                                                 std::optional<bool> &value, std::string &error) {
    return GenericOptionalJsonCheck<bool, OrderedJson::value_t::boolean>(jsonObj, key, value, error);
}

bool ParametersChecker::BooleanJsonCheck(const OrderedJson &jsonObj, const std::string &key, bool &value,
                                         std::string &error) {
    return GenericRequiredJsonCheck<bool, OrderedJson::value_t::boolean>(jsonObj, key, value, error);
}

bool ParametersChecker::OptionalFloatJsonCheck(
    const OrderedJson &jsonObj, const std::string &key, std::optional<float> &output, std::string &error,
    const std::function<bool(const double &, std::stringstream &)> &validator) {
    return NullableJsonCheck<std::optional<float>, double, OrderedJson::value_t::number_float>(jsonObj, key, error,
                                                                                               output, validator);
}

bool ParametersChecker::FloatJsonCheck(const OrderedJson &jsonObj, const std::string &key, float &output,
                                       std::string &error,
                                       const std::function<bool(const float &, std::stringstream &)> &validator) {
    return NullableJsonCheck<float, float, OrderedJson::value_t::number_float>(jsonObj, key, error, output, validator);
}

bool ParametersChecker::Int32JsonCheck(const OrderedJson &jsonObj, const std::string &key, int32_t &output,
                                       std::string &error,
                                       const std::function<bool(const int64_t &, std::stringstream &)> &validator) {
    return NullableJsonCheck<int32_t, int64_t, OrderedJson::value_t::number_integer>(jsonObj, key, error, output,
                                                                                     validator);
}

bool ParametersChecker::OptionalInt32JsonCheck(
    const OrderedJson &jsonObj, const std::string &key, std::optional<int32_t> &output, std::string &error,
    const std::function<bool(const int64_t &, std::stringstream &)> &validator) {
    return NullableJsonCheck<std::optional<int32_t>, int64_t, OrderedJson::value_t::number_integer>(jsonObj, key, error,
                                                                                                    output, validator);
}

bool ParametersChecker::UInt64JsonCheck(const OrderedJson &jsonObj, const std::string &key, uint64_t &output,
                                        std::string &error,
                                        const std::function<bool(const uint64_t &, std::stringstream &)> &validator) {
    return NullableJsonCheck<uint64_t, uint64_t, OrderedJson::value_t::number_unsigned>(jsonObj, key, error, output,
                                                                                        validator);
}

bool ParametersChecker::OptionalUInt64JsonCheck(
    const OrderedJson &jsonObj, const std::string &key, std::optional<uint64_t> &output, std::string &error,
    const std::function<bool(const uint64_t &, std::stringstream &)> &validator) {
    return NullableJsonCheck<std::optional<uint64_t>, uint64_t, OrderedJson::value_t::number_unsigned>(
        jsonObj, key, error, output, validator);
}

bool ParametersChecker::OptionalUint32JsonCheck(
    const OrderedJson &jsonObj, const std::string &key, std::optional<uint32_t> &output, std::string &error,
    const std::function<bool(const uint32_t &, std::stringstream &)> &validator) {
    return NullableJsonCheck<std::optional<uint32_t>, uint32_t, OrderedJson::value_t::number_unsigned>(
        jsonObj, key, error, output, validator);
}
}  // namespace mindie_llm
