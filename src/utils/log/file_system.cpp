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

#include "file_system.h"

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstdint>
#include <cstring>
#include <fstream>

namespace mindie_llm {

using FileStat = struct stat;

constexpr size_t MAX_PATH_LEN = 256;

bool FileSystem::Exists(const std::string &path) {
    FileStat st;
    if (stat(path.c_str(), &st) < 0) {
        return false;
    }
    return true;
}

bool FileSystem::MakeDir(const std::string &dirPath, int mode) {
    int ret = mkdir(dirPath.c_str(), mode);
    return ret == 0;
}

bool FileSystem::Makedirs(const std::string &dirPath, const mode_t mode) {
    uint32_t offset = 0;
    uint32_t pathLen = dirPath.size();
    do {
        const char *str = strchr(dirPath.c_str() + offset, '/');
        offset = (str == nullptr) ? pathLen : str - dirPath.c_str() + 1;
        std::string curPath = dirPath.substr(0, offset);
        if (!Exists(curPath)) {
            if (!MakeDir(curPath, mode)) {
                return false;
            }
        }
    } while (offset != pathLen);
    return true;
}
}  // namespace mindie_llm
