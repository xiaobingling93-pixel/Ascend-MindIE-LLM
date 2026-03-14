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

#ifndef MINDIE_LLM_COMMON_UTIL_H
#define MINDIE_LLM_COMMON_UTIL_H

#include <chrono>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <climits>
#include <iostream>
#include <cstdio>
#include <string>
#include <cctype>
#include <unistd.h>
#include <vector>
#include <map>
#include <sstream>
#include <set>
#include <random>
#include <functional>
#include <cstdint>
#include <sys/stat.h>
#include "nlohmann/json.hpp"
#include "error.h"
#include "file_utils.h"

namespace mindie_llm {
struct LogRotateParam {
    std::string scheduler{"off"};
    uint32_t fs{20};
    uint32_t fc{1};
    uint32_t rotate{10};
};
constexpr uint32_t MAX_CONFIG_FILE_SIZE_LIMIT = 500 * 1024 * 1024; // 500 MB

constexpr int MIN_PRIVATE_KEY_CONTENT_BIT_LEN = 3072; // RSA密钥长度要求大于3072
constexpr int MAX_PRIVATE_KEY_CONTENT_BIT_LEN = 32768; // huge RSA秘钥位宽为32768

constexpr int MIN_PRIVATE_KEY_CONTENT_BYTE_LEN = MIN_PRIVATE_KEY_CONTENT_BIT_LEN / 8; // 1个byte = 8bit
constexpr int MAX_PRIVATE_KEY_CONTENT_BYTE_LEN = MAX_PRIVATE_KEY_CONTENT_BIT_LEN / 8; // 1个byte = 8bit

const mode_t MAX_CONFIG_PERM = S_IRUSR | S_IWUSR | S_IRGRP;   // 640
const mode_t MAX_HOME_DIR_PERM = S_IRWXU | S_IRGRP | S_IXGRP; // 750

std::vector<std::string> GetHostIP(bool skipLoopback = true);

size_t GetDuration(const std::chrono::steady_clock::time_point &end,
                   const std::chrono::steady_clock::time_point &start);

std::string GetCurTime();

std::vector<std::string> Split(const std::string &str, char delim);

std::string TrimSpace(const std::string &str);

std::string ToLower(std::string str);

std::string ToUpper(std::string str);

bool CanonicalPath(std::string &path);

bool GetBinaryPath(std::string &outPath);

Error GetHomePath(std::string &outHomePath);

bool IsNumber(const std::string &str);

Error GetConfigPath(std::string &outConfigPath);

bool GetWorldSizeAndServerCountFromRanktable(size_t &tp, size_t &serverCount);

bool GetModelInfo(std::string &modelName, size_t &tp, size_t &serverCount);

constexpr uint32_t SignalHandlerDefaultTimeout = 5;

void ExecuteAction(std::function<void()> action, uint32_t timeoutSeconds, std::function<void()> timeoutHandler);

template <typename T>
std::string SerializeSet(const std::set<T>& inputSet)
{
    std::stringstream ss;
    bool first = true;
    for (T elem : inputSet) {
        if (!first) {
            ss << ","; // 使用逗号作为分隔符
        }
        first = false;
        ss << elem;
    }
    return ss.str();
}

std::set<size_t> DeserializeSet(const std::string &data);

std::string JoinStrings(const std::vector<std::string> &stringsVec, const std::string &delimiter);

uint32_t RandomNumber(uint32_t maxNumber);

bool CheckAndGetLogPath(const std::string &logPath, uint64_t sizeLimit, std::string &outPath,
                        const std::string &defaultPath);

std::vector<std::string> SplitPath(const std::string &absPath) noexcept;

std::string AbsoluteToAnonymousPath(const std::string &absPath) noexcept;

std::string AbsoluteToRelativePath(const std::string &absPath, const std::string &absDir) noexcept;

std::string CleanStringForJson(const std::string &input);

bool IsFloatEquals(float a, float b);

template <typename T>
inline void StreamAppend(std::stringstream &stream, typename std::vector<T> source,
                         size_t limit = std::numeric_limits<size_t>::max(), bool delimOnStart = false,
                         const std::string &delim = ",")
{
    limit = std::min(source.size(), limit);
    if (limit == 0) {
        return;
    }

    stream << (delimOnStart ? delim : "") << source[0];

    for (size_t i = 1; i < limit; ++i) {
        stream << delim << source[i];
    }
}

std::vector<std::string> SplitString(const std::string &str, char delimiter);

bool CheckSystemConfig(const std::string &jsonPath, nlohmann::json &inputJsonData, std::string paramType);

bool ReadJsonFile(const std::string &jsonPath, std::string &baseDir, nlohmann::json &inputJsonData,
                  std::string paramType);

void GetModelInfo(const std::string &configPath, std::string &modelName, size_t &tp, size_t &serverCount);

Error GetLlmPath(std::string &outHomePath);

bool ParsePortFromIp(const std::string& ipPort, uint32_t& port);

std::string JoinStrings(const std::vector<std::string> &stringsVec, const std::string &delimiter);

std::pair<uint32_t, uint32_t> ReverseDpInstId(uint64_t dpInstanceId);

bool CheckIp(const std::string &ipAddress, const std::string &inputName, bool enableZeroIp);

bool CheckIPV4(const std::string &ipAddress, const std::string &inputName, bool enableZeroIp);

bool CheckIPV6(const std::string &ipAddress, const std::string &inputName, bool enableZeroIp);

bool IsIPv4(const std::string &ipAddress);

bool IsIPv6(const std::string &ipAddress);

std::string FormatGrpcAddress(const std::string& ip, const std::string& port);

// safe get value from map<vector<int64_t>>
bool SafeGetMapVectorValue(const std::map<uint64_t, std::vector<int64_t>>& map,
                           uint64_t seqId,
                           size_t index,
                           int64_t& outValue,
                           const std::string& mapName) noexcept;

// safe get value from map<vector<float>>
bool SafeGetMapVectorValue(const std::map<uint64_t, std::vector<float>>& map,
                           uint64_t seqId,
                           size_t index,
                           float& outValue,
                           const std::string& mapName) noexcept;

bool StrToInt64(int64_t &dest, const std::string &str);

bool StrToUint64(uint64_t &dest, const std::string &str);

bool StrToUint32(uint32_t &dest, const std::string &str);

template <typename T>
std::string VectorToString(const std::vector<T>& vec)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        oss << vec[i];
        if (i < vec.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

// Helper function to convert a std::map to a string
template <typename K, typename V>
std::string MapToString(const std::map<K, V>& map)
{
    std::ostringstream oss;
    oss << "{";
    for (auto it = map.begin(); it != map.end(); ++it) {
        oss << it->first << ": " << it->second;
        if (std::next(it) != map.end()) {
            oss << ", ";
        }
    }
    oss << "}";
    return oss.str();
}

// Specialization for std::map with std::vector as the value type
template <typename K, typename V>
std::string MapToString(const std::map<K, std::vector<V>>& map)
{
    std::ostringstream oss;
    oss << "{";
    for (auto it = map.begin(); it != map.end(); ++it) {
        oss << it->first << ": " << VectorToString(it->second);
        if (std::next(it) != map.end()) {
            oss << ", ";
        }
    }
    oss << "}";
    return oss.str();
}

template <typename KeyType, typename ValueType>
void MergeMaps(std::map<KeyType, ValueType>& totalMap, const std::map<KeyType, ValueType>& subMap)
{
    for (const auto& entry : subMap) {
        totalMap[entry.first] += entry.second;
    }
}

template <typename K, typename V>
std::map<K, V> RemoveMapElements(const std::map<K, V>& inputMap, const std::vector<K>& keysToRemove)
{
    std::map<K, V> resultMap = inputMap;

    for (const auto& key : keysToRemove) {
        // 删除整个键值对
        resultMap.erase(key);
    }
    return resultMap;
}
} // namespace mindie_llm

#endif // MINDIE_LLM_COMMON_UTIL_H