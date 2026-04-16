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

#ifndef MINDIE_LLM_UTILS_FILESYSTEM_H
#define MINDIE_LLM_UTILS_FILESYSTEM_H
#include <string>

namespace mindie_llm {
class FileSystem {
   public:
    static bool Exists(const std::string &path);
    static bool MakeDir(const std::string &dirPath, int mode);
    static bool Makedirs(const std::string &dirPath, const mode_t mode);
};
}  // namespace mindie_llm
#endif
