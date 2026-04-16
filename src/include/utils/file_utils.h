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

#ifndef MINDIE_LLM_FILE_UTIL_H
#define MINDIE_LLM_FILE_UTIL_H

#include <cstdint>
#include <string>

namespace mindie_llm {
struct FileValidationParams {
    bool isFileExist;
    mode_t mode;
    uint64_t maxfileSize;
    bool checkPermission;
};
class FileUtils {
   public:
    static const mode_t FILE_MODE_750 = 0b111'101'000;
    static const mode_t FILE_MODE_550 = 0b101'101'000;
    static const mode_t FILE_MODE_500 = 0b101'000'000;
    static const mode_t FILE_MODE_640 = 0b110'100'000;
    static const mode_t FILE_MODE_600 = 0b110'000'000;
    static const mode_t FILE_MODE_440 = 0b100'100'000;
    static const mode_t FILE_MODE_400 = 0b100'000'000;
    /**
     * judge file exists
     * @param path: file full path
     * @param pattern: regex pattern
     */
    static bool CheckFileExists(const std::string &filePath);

    /**
     * is directory exists.
     * @param dir directory
     * @return
     */
    static bool CheckDirectoryExists(const std::string &dirPath);

    /** Check whether the destination path is a link
     * @param filePath raw file path
     * @return
     */
    static bool IsSymlink(const std::string &filePath);

    /** Regular the file path using realPath.
     * @param filePath file path
     * @param baseDir file path must in base dir
     * @param errMsg the err msg
     * @return
     */
    static bool RegularFilePath(const std::string &filePath, const std::string &baseDir, std::string &errMsg,
                                std::string &regularPath);

    /** Regular the file path using realPath.
     * @param filePath raw file path
     * @param baseDir file path must in base dir
     * @param errMsg the err msg
     * @return
     */
    static bool RegularFilePath(const std::string &filePath, const std::string &baseDir, std::string &errMsg, bool flag,
                                std::string &regularPath);

    /** Regular the file path using realPath.
     * @param filePath raw file path
     * @param errMsg the err msg
     * @return
     */
    static bool RegularFilePath(const std::string &filePath, std::string &errMsg, std::string &regularPath);

    /** Check the existence of the file and the size of the file.
     * @param filePath the input file path
     * @param errMsg the err msg
     * @param isFileExit isFileExit if true, then file not exit, return false; else return true
     * @param mode file mode such as 0b111'101'000 means 750
     * @param checkPermission checkPermission if true, check file mode; else not check
     * @param maxfileSize default 10485760(10M)， maxfileSize must in (1, 1073741824](1G)
     * @return
     */
    static bool IsFileValid(const std::string &filePath, std::string &errMsg, bool isFileExit = true,
                            mode_t mode = FILE_MODE_750, bool checkPermission = true, uint64_t maxfileSize = 10485760);

    /** Check the existence of the file and the size of the file.
     * @param configFile the input file path
     * @param errMsg the err msg
     * @param checkPermission check perm
     * @param onlyCurrentUserOp strict check, only current user can write or execute
     * @return
     */
    static bool IsFileValid(const std::string &filePath, std::string &errMsg, const FileValidationParams &params);
    static bool IsFileAndDirectoryExists(const std::string &filePath, std::string &errMsg,
                                         const FileValidationParams &params);
    static bool GetCheckPermissionFlag();
    /** Check the file owner, file must be owner current user
     * @param filePath the input file path
     * @param errMsg error msg
     * @return
     */
    static bool ConstrainOwner(const std::string &filePath, std::string &errMsg);

    /** Check the file mode, file must be no greater than mode
     * @param filePath the input file path
     * @param mode file mode
     * @param errMsg error msg
     * @return
     */
    static bool ConstrainPermission(const std::string &filePath, const mode_t &mode, std::string &errMsg);

    /** Check the existence of the file and the size of the file.
     * @param filePath the input file path
     * @param realFilePath the real file path
     * @param errMsg the err msg
     * @return
     */
    static bool GetRealFilePath(const std::string &filePath, std::string &realFilePath, std::string &errMsg) noexcept;

    /** Check the existence of the file and the size of the file.
     * @param installPath the install path
     * @param errMsg the err msg
     * @return
     */
    static bool GetInstallPath(std::string &installPath, std::string &errMsg) noexcept;

    /**
     * @brief Get the safe relative path.
     *
     * @param oldPath input path
     * @param safePath output safe path
     * @return
     */
    static std::string GetSafeRelativePath(const std::string &oldPath);
};

}  // namespace mindie_llm

#endif  // MINDIE_LLM_FILE_UTIL_H
