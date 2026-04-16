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

#include "log_utils.h"

#include <cstdio>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>

#include "file_utils.h"
#include "log_config.h"

namespace mindie_llm {

void LogUtils::SetMindieLogParamBool(LoggerType loggerType, bool &logParam, const std::string &envVar) {
    std::string envParam = LogUtils::GetEnvParam(loggerType, envVar);
    std::transform(envParam.begin(), envParam.end(), envParam.begin(), ::tolower);

    const std::unordered_set<std::string> validBoolKeys = {"true", "false", "1", "0"};
    static std::unordered_map<std::string, bool> logBoolMap = {
        {"0", false},
        {"1", true},
        {"false", false},
        {"true", true},
    };

    if (validBoolKeys.find(envParam) != validBoolKeys.end()) {
        logParam = logBoolMap[envParam];
    }
}

void LogUtils::SetMindieLogParamString(LoggerType loggerType, std::string &logParam, const std::string &envVar) {
    std::string envParam = LogUtils::GetEnvParam(loggerType, envVar);
    if (!envParam.empty()) {
        logParam = envParam;
    }
}

void LogUtils::SetMindieLogParamLevel(LoggerType loggerType, LogLevel &logParam, const std::string &envVar) {
    std::string envParam = LogUtils::GetEnvParam(loggerType, envVar);
    std::transform(envParam.begin(), envParam.end(), envParam.begin(), ::tolower);

    const std::unordered_set<std::string> validLevelKeys = {"debug", "info", "warn", "error", "critical"};
    static std::unordered_map<std::string, LogLevel> logLevelMap = {
        {"debug", LogLevel::debug}, {"info", LogLevel::info},         {"warn", LogLevel::warn},
        {"error", LogLevel::err},   {"critical", LogLevel::critical},
    };

    if (validLevelKeys.find(envParam) != validLevelKeys.end()) {
        logParam = logLevelMap[envParam];
    }
}

std::string LogUtils::GetEnvParam(LoggerType loggerType, const std::string &mindieEnv) {
    std::vector<std::string> modules = LogUtils::Split(mindieEnv, ';');
    std::string flag;
    const std::string &loggerModule = GetModuleName(loggerType);
    for (auto &module : modules) {
        module = LogUtils::Trim(module);
        size_t colonPos = module.find(':');
        if (colonPos != std::string::npos) {
            std::string moduleName = module.substr(0, colonPos);
            moduleName = LogUtils::Trim(moduleName);
            if (moduleName == loggerModule || moduleName == ALL_COMPONENT_NAME) {
                flag = module.substr(colonPos + 1);
                flag = LogUtils::Trim(flag);
            }
        } else {
            flag = module;
        }
    }
    return flag;
}

std::string LogUtils::Trim(std::string str) {
    if (str.empty()) {
        std::cout << "str is empty." << std::endl;
        return str;
    }

    str.erase(0, str.find_first_not_of(" "));
    str.erase(str.find_last_not_of(" ") + 1);
    return str;
}

std::vector<std::string> LogUtils::Split(const std::string &str, char delim) {
    std::vector<std::string> tokens;
    // 1. check empty string
    if (str.empty()) {
        std::cout << "str is empty." << std::endl;
        return tokens;
    }

    auto stringFindFirstNot = [str, delim](size_t pos = 0) -> size_t {
        for (size_t i = pos; i < str.size(); i++) {
            if (str[i] != delim) {
                return i;
            }
        }
        return std::string::npos;
    };

    size_t lastPos = stringFindFirstNot(0);
    size_t pos = str.find(delim, lastPos);
    while (lastPos != std::string::npos) {
        tokens.emplace_back(str.substr(lastPos, pos - lastPos));
        lastPos = stringFindFirstNot(pos);
        pos = str.find(delim, lastPos);
    }
    return tokens;
}

void LogUtils::UpdateLogFileParam(std::string rotateConfig, uint32_t &maxFileSize, uint32_t &maxFiles) {
    if (rotateConfig.empty()) {
        return;
    }
    std::istringstream configStream(rotateConfig);
    std::string option;
    std::string value;

    auto isNumeric = [](const std::string &str) {
        return !str.empty() && std::all_of(str.begin(), str.end(), ::isdigit);
    };

    while (configStream >> option) {
        if (!(configStream >> value)) {
            continue;
        }
        if (option == "-fs" && isNumeric(value)) {
            maxFileSize = static_cast<uint32_t>(std::stoi(value)) * 1024 * 1024;  // 1 MB = 1024 KB = 1024 * 1024 B;
            if (maxFileSize > LOG_FILE_SIZE_LIMIT) {
                throw std::runtime_error("log file size should not be set bigger than" +
                                         std::to_string(LOG_FILE_SIZE_LIMIT));
            }
        } else if (option == "-r" && isNumeric(value)) {
            maxFiles = static_cast<uint32_t>(std::stoi(value));
            if (maxFiles > LOG_FILE_NUM_LIMIT) {
                throw std::runtime_error("log file num should not be set bigger than" +
                                         std::to_string(LOG_FILE_NUM_LIMIT));
            }
        }
    }
}

void LogUtils::GetLogFileName(LoggerType loggerType, std::string &filename) {
    int pid = spdlog::details::os::pid();
    auto now = std::chrono::system_clock::now();

    auto duration = now.time_since_epoch();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
    std::tm nowTm = *std::localtime(&nowTime);

    std::stringstream ss;
    ss << std::put_time(&nowTm, "%Y%m%d%H%M%S");

    int millisecondsPart = milliseconds % 1000;

    const uint32_t millisecondsWidth = 3;
    ss << std::setw(millisecondsWidth) << std::setfill('0') << millisecondsPart;

    std::string timeStr = ss.str();
    filename += "/mindie-" + GetLoggerNameStr(loggerType) + "_" + std::to_string(pid) + "_" + timeStr + ".log";
}

std::string LogUtils::GetModuleName(LoggerType loggerType) {
    loggerType = (loggerType == LoggerType::MINDIE_LLM_REQUEST || loggerType == LoggerType::MINDIE_LLM_TOKEN)
                     ? LoggerType::MINDIE_LLM
                     : loggerType;
    auto it = MODULE_NAME_MAP.find(loggerType);
    return it != MODULE_NAME_MAP.end() ? it->second : throw std::invalid_argument("Invalid LoggerType enum value");
}

std::string LogUtils::GetLoggerNameStr(LoggerType loggerType) {
    auto it = LOGGER_NAME_MAP.find(loggerType);
    return it != LOGGER_NAME_MAP.end() ? it->second : throw std::invalid_argument("Invalid LoggerType enum value");
}

GenericRotationFileSink::GenericRotationFileSink(const std::string &baseFileName, size_t maxFileSize, size_t maxFileNum,
                                                 const spdlog::file_event_handlers &eventHandlers,
                                                 const std::string &baseDir)
    : baseFileName_(baseFileName),
      maxFileSize_(maxFileSize),
      maxFileNum_(maxFileNum),
      mtx_(),
      fileHelper_(std::make_unique<spdlog::details::file_helper>(eventHandlers)),
      isFileCreated_(std::experimental::filesystem::exists(baseFileName_)),
      currentSize_(0),
      lastError_(),
      baseDir_(baseDir) {}

GenericRotationFileSink::~GenericRotationFileSink() = default;

bool GenericRotationFileSink::CreateFileIfNeeded() {
    if (!isFileCreated_) {
        std::string errMsg;
        std::string regularPath;

        try {
            int fd = open(baseFileName_.c_str(), O_WRONLY | O_CREAT | O_EXCL, MAX_OPEN_LOG_FILE_PERM);
            if (fd == -1) {
                throw std::runtime_error("Creating log file error: " + baseFileName_);
            }
            close(fd);

            FileValidationParams fileParams = {false, MAX_OPEN_LOG_FILE_PERM, MAX_ROTATION_FILE_SIZE_LIMIT, true};
            if (!FileUtils::RegularFilePath(baseFileName_, baseDir_, errMsg, true, regularPath) ||
                !FileUtils::IsFileValid(regularPath, errMsg, fileParams)) {
                std::cerr << errMsg << std::endl;
                return false;
            }

            fileHelper_->open(regularPath, std::ios_base::app);

            isFileCreated_ = true;
            lastError_.clear();
            return true;
        } catch (const std::exception &e) {
            lastError_ = "Failed to open " + baseFileName_ + ", error: " + e.what();
            return false;
        }
    }
    return true;
}

const std::string &GenericRotationFileSink::GetLastError() const { return lastError_; }

void GenericRotationFileSink::sink_it_(const spdlog::details::log_msg &msg) {
    if (!CreateFileIfNeeded()) {
        return;
    }

    if (currentSize_ == 0) {
        currentSize_ = fileHelper_->size();
    }

    spdlog::memory_buf_t formattedMsg;
    base_sink<std::mutex>::formatter_->format(msg, formattedMsg);

    size_t curSize = currentSize_ + formattedMsg.size();
    if (curSize > maxFileSize_) {
        fileHelper_->flush();
        if (fileHelper_->size() > 0) {
            Rotate();
            curSize = formattedMsg.size();
        }
    }
    fileHelper_->write(formattedMsg);
    fileHelper_->flush();
    currentSize_ = curSize;
}

void GenericRotationFileSink::flush_() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (isFileCreated_) {
        fileHelper_->flush();
    }
}

std::string GenericRotationFileSink::GenerateFileName(std::string &fileName, size_t index) const {
    if (index == 0u) {
        return fileName;
    }
    std::string baseName;
    std::string extName;
    std::tie(baseName, extName) = spdlog::details::file_helper::split_by_extension(fileName);
    return spdlog::fmt_lib::format(SPDLOG_FMT_STRING(SPDLOG_FILENAME_T("{}.{:02}{}")), baseName, index, extName);
}

bool GenericRotationFileSink::RenameFile(std::string &srcFileName, std::string &targetFileName) const {
    (void)spdlog::details::os::remove(targetFileName);
    return spdlog::details::os::rename(srcFileName, targetFileName) == 0;
}

void GenericRotationFileSink::Rotate() {
    using spdlog::details::os::filename_to_str;
    using spdlog::details::os::path_exists;

    fileHelper_->close();
    for (auto i = maxFileNum_; i > 0; --i) {
        std::string src = GenerateFileName(baseFileName_, i - 1);
        if (!path_exists(src)) {
            continue;
        }
        std::string target = GenerateFileName(baseFileName_, i);
        if (!RenameFile(src, target)) {
            // retry
            spdlog::details::os::sleep_for_millis(INTERVAL_OF_SLEEP);
            if (!RenameFile(src, target)) {
                fileHelper_->reopen(true);
                currentSize_ = 0;
                std::cerr << "Error: Failed to rename " + filename_to_str(src) + " to " + filename_to_str(target);
            }
        }
    }
    fileHelper_->reopen(true);
}

}  // namespace mindie_llm
