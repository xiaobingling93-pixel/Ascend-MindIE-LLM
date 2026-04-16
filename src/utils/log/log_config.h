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

#ifndef MINDIE_LLM_LOG_CONFIG_H
#define MINDIE_LLM_LOG_CONFIG_H

#include <sys/stat.h>

#include "nlohmann/json.hpp"
#include "spdlog.h"

namespace mindie_llm {

using Json = nlohmann::json;
using LogLevel = spdlog::level::level_enum;

// Default log settings
const bool DEFAULT_LOG_TO_STDOUT = false;
const bool DEFAULT_LOG_TO_STDOUT_ATB = true;
const bool DEFAULT_LOG_TO_FILE = true;
const LogLevel DEFAULT_LOG_LEVEL = LogLevel::info;
const bool DEFAULT_LOG_VERBOSE = true;
const std::string DEFAULT_LOG_PATH = "mindie/log";
constexpr uint32_t DEFAULT_LOG_FILE_COUNT = 10U;
constexpr uint32_t DEFAULT_LOG_FILE_SIZE = 20U;  // 20 MB

extern const char* MODULE_NAME_LLM;
extern const char* MODULE_NAME_ATB;
extern const char* MODULE_NAME_SERVER;
extern const char* LOGGER_NAME_LLM_TOKEN;
extern const char* LOGGER_NAME_LLM_REQUEST;
extern const char* ALL_COMPONENT_NAME;
// Log setting limits
constexpr size_t MAX_PATH_LENGTH = 4096;
constexpr uint32_t MAX_ROTATION_FILE_COUNT_LIMIT = 64;
constexpr uint32_t MIN_ROTATION_FILE_COUNT_LIMIT = 1;
constexpr uint32_t MAX_ROTATION_FILE_SIZE_LIMIT = 500 * 1024 * 1024;  // 500 MB
constexpr uint32_t MIN_ROTATION_FILE_SIZE_LIMIT = 1 * 1024 * 1024;    // 1 MB
const mode_t MAX_LOG_DIR_PERM = S_IRWXU | S_IRGRP | S_IXGRP;          // 750
const mode_t MAX_OPEN_LOG_FILE_PERM = S_IRUSR | S_IWUSR | S_IRGRP;    // 640
const mode_t MAX_CLOSE_LOG_FILE_PERM = S_IRUSR | S_IRGRP;             // 440
constexpr int MAX_LOG_LEVEL_LIMIT = 5;

constexpr int LOG_OK = 0;
constexpr int LOG_INVALID_PARAM = 3;

const size_t LOGGER_QUEUE_SIZE = 8192;  // For async logger, this's queue size
const size_t LOGGER_THREAD_NUM = 1;     // For async logger, we can specify number of threads

const std::unordered_map<std::string, LogLevel> LOG_LEVEL_MAP{
    {"DEBUG", LogLevel::debug},  {"INFO", LogLevel::info}, {"WARN", LogLevel::warn},
    {"WARNING", LogLevel::warn}, {"ERROR", LogLevel::err}, {"CRITICAL", LogLevel::critical},
};

enum class LoggerType { MINDIE_LLM, MINDIE_LLM_REQUEST, MINDIE_LLM_TOKEN, ATB, SECURITY, DEBUG, MAX_LOGGER_TYPE };
extern const std::unordered_map<LoggerType, std::string> LOGGER_NAME_MAP;
extern const std::unordered_map<LoggerType, std::string> MODULE_NAME_MAP;

class LogConfig {
   public:
    LogConfig() = default;
    LogConfig(const LogConfig& config);
    LogConfig& operator=(const LogConfig& config) = delete;
    ~LogConfig() = default;

    int Init(LoggerType loggerType);
    int ValidateSettings();
    void MakeDirsWithTimeOut(const std::string& parentPath) const;

   public:
    bool logToStdOut_ = DEFAULT_LOG_TO_STDOUT;
    bool logToFile_ = DEFAULT_LOG_TO_FILE;
    bool logVerbose_ = DEFAULT_LOG_VERBOSE;
    LogLevel logLevel_ = DEFAULT_LOG_LEVEL;
    std::string baseDir_;
    std::string logFilePath_;
    std::string logRotateConfig_;
    uint32_t logFileSize_ = DEFAULT_LOG_FILE_SIZE * 1024 * 1024;  // 1 MB = 1024 KB = 1024 * 1024 B
    uint32_t logFileCount_ = DEFAULT_LOG_FILE_COUNT;

   private:
    void InitLogToStdoutFlag(LoggerType loggerType);
    void InitLogToFileFlag(LoggerType loggerType);
    void InitLogLevel(LoggerType loggerType);
    void InitLogFilePath(LoggerType loggerType);
    void InitLogVerbose(LoggerType loggerType);
    void InitLogRotationParam(LoggerType loggerType);
    bool CheckAndGetLogPath(const std::string& configLogPath);
};

}  // namespace mindie_llm

#endif  // MINDIE_LLM_LOG_CONFIG_H
