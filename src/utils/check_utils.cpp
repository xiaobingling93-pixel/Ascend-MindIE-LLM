/**
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

#include "check_utils.h"

#include <iostream>
#include <map>
#include <regex>

#include "basic_types.h"

namespace mindie_llm {
// Param Type Size
const size_t PACK_QUANT_TYPE_LENGTH = 2;
const size_t LINEAR_TYPE_LENGTH = 7;
const int MAX_NUM_HIDDEN_LAYER = 1000;

static std::map<std::string, std::pair<std::string, std::string>> g_integerTypeMap = {
    {"int32_t", {"2147483647", "-2147483648"}},
    {"uint32_t", {"4294967295", "0"}},
    {"size_t", {"18446744073709551615", "0"}},
};

int CheckParamRange(const int& intParam, int min, int max) {
    if (intParam < min) {
        std::stringstream ss;
        ss << "This param must be a number greater or equal to " << min << ", please check." << std::endl;
        throw std::runtime_error(ss.str());
    }
    if (intParam > max) {
        std::stringstream ss;
        ss << "This param must be a number less or equal to " << max << ", please check." << std::endl;
        throw std::runtime_error(ss.str());
    }
    return intParam;
}

int CheckPositive(const int& intParam) {
    if (intParam <= 0) {
        std::stringstream ss;
        ss << "This param must be a number greater than 0, please check." << std::endl;
        throw std::runtime_error(ss.str());
    }
    return intParam;
}

std::string NormalizeIntegerString(const std::string& s) {
    if (s.empty()) {
        return s;
    }
    bool isNegative = (s[0] == '-');
    bool hasSign = (s[0] == '-' || s[0] == '+');
    size_t start = hasSign ? 1 : 0;

    // 跳过前导 0
    size_t i = start;
    while (i < s.size() && s[i] == '0') {
        i++;
    }

    // 全是 0 -> 归一为 "0"
    if (i == s.size()) {
        return "0";
    }
    std::string core = s.substr(i);
    return isNegative ? "-" + core : core;
}

bool IsWithinRange(const std::string& integerType, const Json& jsonValue) {
    std::string value = NormalizeIntegerString(jsonValue.dump());
    bool isIntType = (integerType[0] == 'i');
    std::string maxValue = g_integerTypeMap[integerType].first;
    std::string minValue = g_integerTypeMap[integerType].second;
    size_t valueLength = value.length();
    size_t minValLength = minValue.length();
    size_t maxValLength = maxValue.length();

    if (isIntType) {
        if (value.find('-') != std::string::npos) {
            return (!(valueLength > minValLength || (valueLength == minValLength && value.compare(minValue) > 0)));
        } else {
            return (!(valueLength > maxValLength || (valueLength == maxValLength && value.compare(maxValue) > 0)));
        }
    } else {
        if (valueLength > maxValLength || (value.find('-') != std::string::npos) ||
            (valueLength == maxValLength && value.compare(maxValue) > 0)) {
            return false;
        }
    }

    return true;
}

uint32_t GetIntegerParamValue(const Json& jsonData, const std::string& configName, uint32_t defaultVal) {
    if (jsonData.empty() || !jsonData.contains(configName)) {
        return defaultVal;
    }
    if (!jsonData[configName].is_number_integer()) {
        std::cout << "The type of [" << configName << "] should be integer; the default value is applied." << std::endl;
        return defaultVal;
    }
    if (!IsWithinRange("uint32_t", jsonData[configName])) {
        std::cout << "The value of [" << configName << "] is out of range as uint32_t; the default value is applied."
                  << std::endl;
        return defaultVal;
    }
    return jsonData[configName];
}

std::string GetStringParamValue(const Json& jsonData, const std::string& configName) {
    if (!jsonData.contains(configName)) {
        throw std::runtime_error("Config parameter '" + configName + "' not found");
    }

    const auto& value = jsonData[configName];
    if (!value.is_string()) {
        throw std::runtime_error("Config parameter '" + configName + "' is not a string");
    }

    std::string res = value.get<std::string>();
    if (res.length() > MAX_STRING_LENGTH) {
        throw std::runtime_error("The length of config parameter '" + configName +
                                 "' exceeds the maximum limit: " + std::to_string(MAX_STRING_LENGTH));
    }

    return res;
}

bool CheckStringInputLength(const std::string& stringValue, uint32_t maxValLength) {
    if (stringValue.length() > maxValLength) {
        return false;
    }
    return true;
}
}  // namespace mindie_llm
