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

#ifndef MINDIE_LLM_LOG_UTILS_H
#define MINDIE_LLM_LOG_UTILS_H

#include <cstddef>
#include <ios>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "log_config.h"
#include "spdlog.h"

namespace mindie_llm {

const size_t INTERVAL_OF_SLEEP = 100;
const size_t LOG_FILE_SIZE_LIMIT = 524288000;
const size_t LOG_FILE_NUM_LIMIT = 64;

class LogUtils {
   public:
    static void SetMindieLogParamBool(LoggerType loggerType, bool &logParam, const std::string &envVar);

    static void SetMindieLogParamString(LoggerType loggerType, std::string &logParam, const std::string &envVar);

    static void SetMindieLogParamLevel(LoggerType loggerType, LogLevel &logParam, const std::string &envVar);

    static std::string GetEnvParam(LoggerType loggerType, const std::string &mindieEnv);

    static std::string Trim(std::string str);

    static std::vector<std::string> Split(const std::string &str, char delim = ' ');

    static void UpdateLogFileParam(std::string rotateConfig, uint32_t &maxFileSize, uint32_t &maxFiles);

    static void GetLogFileName(LoggerType loggerType, std::string &filename);

    static std::string GetModuleName(LoggerType loggerType);

    static std::string GetLoggerNameStr(LoggerType loggerType);
};

class GenericRotationFileSink : public spdlog::sinks::base_sink<std::mutex> {
   public:
    GenericRotationFileSink(const std::string &baseFileName, size_t maxFileSize, size_t maxFileNum,
                            const spdlog::file_event_handlers &eventHandlers, const std::string &baseDir);
    ~GenericRotationFileSink() override;

    bool CreateFileIfNeeded();
    const std::string &GetLastError() const;

   protected:
    void sink_it_(const spdlog::details::log_msg &msg) override;
    void flush_() override;

   private:
    std::string GenerateFileName(std::string &fileName, size_t index) const;

    bool RenameFile(std::string &srcFileName, std::string &targetFileName) const;

    void Rotate();

    std::string baseFileName_;
    size_t maxFileSize_;
    size_t maxFileNum_;
    mutable std::mutex mtx_;
    std::unique_ptr<spdlog::details::file_helper> fileHelper_;
    bool isFileCreated_;
    size_t currentSize_;
    std::string lastError_;
    std::string baseDir_;
};

}  // namespace mindie_llm

#endif  // MINDIE_LLM_LOG_UTILS_H
