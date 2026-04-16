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

#include "shared_memory.h"

#include <sys/mman.h>
#include <sys/statfs.h>

#include <algorithm>
#include <cstring>

#include "file_utils.h"
#include "log.h"
#include "memory_utils.h"
constexpr mode_t FULL_PERMISSION_MASK = 0777;
constexpr mode_t REQUIRED_PERMISSION = 0600;
namespace mindie_llm {
static constexpr uint32_t MEM_PAGE_SIZE = 4096U;

SharedMemory::~SharedMemory() {
    if (mMapBuf != nullptr) {
        munmap(mMapBuf, mCurSize);
        mMapBuf = nullptr;
    }
    if (mFd_ > 0) {
        close(mFd_);
        mFd_ = 0;
        shm_unlink(this->mName.c_str());  // 移除共享内存, 其他进程无法访问
    }
    valid = false;
}

bool SharedMemorySizeCheck(const uint64_t &pendingMemoryAllocationSize) {
    const std::string path = "/dev/shm";

    // check path exists or not
    if (!FileUtils::CheckDirectoryExists(path)) {
        MINDIE_LLM_LOG_ERROR("Shared memory directory does not exist for path %s.", path);
        return false;
    }

    // check path is a link or not
    if (FileUtils::IsSymlink(path)) {
        MINDIE_LLM_LOG_ERROR("Shared memory path is symlink for path %s.", path);
        return false;
    }

    struct statfs buf;
    // get filesystem information by statfs function
    if (statfs(path.c_str(), &buf) == -1) {
        MINDIE_LLM_LOG_ERROR("Failed to get shared memory file system information for path %s.", path);
        return false;
    }

    // available size of the shared memory filesystem
    uint64_t availSize = static_cast<uint64_t>(buf.f_bsize) * buf.f_bavail;

    if (availSize < pendingMemoryAllocationSize) {
        MINDIE_LLM_LOG_ERROR(
            "Shared memory available is not enough on the filesystem with "
            " available size "
            << availSize << " and pending allocation size " << pendingMemoryAllocationSize);
        return false;
    }
    return true;
}

bool SharedMemory::SharedMemoryNameChecker(const std::string &name) {
    if (name.empty() || name.length() > LLM_SHARED_MEMORY_MAX_NAME_LEN) {
        return false;
    }

    // 第一个字符为“/”，后续没有“/”
    if (name[0] != '/') {
        return false;
    } else if (count(name.begin(), name.end(), '/') != 1) {
        return false;
    }
    return true;
}

bool SharedMemory::SharedMemoryUIDAndPermissionChecker(FileDesc mFd) {
    struct stat shm_stat;
    if (fstat(mFd, &shm_stat) != 0) {
        MINDIE_LLM_LOG_ERROR("Failed to fstat shared memory " << this->mName);
        close(mFd);
        return false;
    }
    uid_t current_uid = getuid();
    if (shm_stat.st_uid != current_uid) {
        MINDIE_LLM_LOG_ERROR("Shared memory " << this->mName << " owned by uid " << shm_stat.st_uid
                                              << ", but current uid is " << current_uid);
        close(mFd);
        return false;
    }
    if ((shm_stat.st_mode & FULL_PERMISSION_MASK) != REQUIRED_PERMISSION) {
        MINDIE_LLM_LOG_ERROR("Shared memory " << this->mName << " permission expected 0600");
        close(mFd);
        return false;
    }
    return true;
}

bool SharedMemory::Create(const std::string &name, uint32_t size) {
    this->mName = name;
    this->mCurSize = size;
    if (!SharedMemorySizeCheck(size)) {
        MINDIE_LLM_LOG_ERROR("Available shared memory size is not enough with name " << name << ", size " << size);
        return false;
    }
    if (!SharedMemoryNameChecker(name)) {
        MINDIE_LLM_LOG_ERROR("The shared memory name format is abnormal with name " << name);
        return false;
    }
    mFd_ = shm_open(name.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    if (mFd_ < 0 || !SharedMemoryUIDAndPermissionChecker(mFd_)) {
        MINDIE_LLM_LOG_ERROR("Failed to open shared memory with name " << name);
        return false;
    }
    if (ftruncate(mFd_, size) == -1) {
        MINDIE_LLM_LOG_ERROR("Failed to ftruncate the shared memory with size " << size);
        close(mFd_);
        return false;
    }
    mMapBuf = (char *)mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, mFd_, 0);
    if (mMapBuf == MAP_FAILED) {
        close(mFd_);
        mFd_ = 0;
        MINDIE_LLM_LOG_ERROR("Failed to mmap shared memory with size " << size);
        return false;
    }
    // 确保内存物理页被分配
    for (auto pos = 0U; pos < mCurSize; pos += MEM_PAGE_SIZE) {
        mMapBuf[pos] = '\0';
    }

    close(mFd_);
    mFd_ = 0;
    valid = true;
    return true;
}

bool SharedMemory::Write(uint32_t dstOffset, const char *src, uint32_t size) const {
    if (!valid) {
        MINDIE_LLM_LOG_ERROR("Shared memory allocation is invalid.");
        return false;
    }
    if (src == nullptr) {
        MINDIE_LLM_LOG_ERROR("Src is invalid.");
        return false;
    }
    if (UINT32_MAX - size < dstOffset) {
        MINDIE_LLM_LOG_ERROR("Integer overflow: " << size << " + " << dstOffset);
        return false;
    }

    if (dstOffset + size > this->mCurSize) {
        MINDIE_LLM_LOG_ERROR("No enough shared memory, try lower request length with size " << size);
        return false;
    }
    int ret = memcpy_s(mMapBuf + dstOffset, this->mCurSize - dstOffset, src, size);
    if (ret != 0) {
        MINDIE_LLM_LOG_ERROR("Failed to memory copy with size " << size);
        return false;
    }
    return true;
}

char *SharedMemory::GetBuf() { return mMapBuf; }

char *SharedMemory::GetBufEnd() const { return mMapBuf + mCurSize; }

int SharedMemory::GetFd() const { return mFd_; }
}  // namespace mindie_llm
