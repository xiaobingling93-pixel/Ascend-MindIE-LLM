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

#ifndef THREAD_GROUP_CC_H
#define THREAD_GROUP_CC_H

#include <algorithm>
#include <any>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

namespace mindie_llm {
// 集合通信类型
enum class CollectiveType : int8_t {
    BARRIER,     // 屏障同步
    BROADCAST,   // 广播
    GATHER,      // 收集
    SCATTER,     // 分发
    ALL_TO_ALL,  // 全交换
    REDUCE       // 归约
};

// 归约操作类型
enum class ReduceOp {
    SUM,      // 求和
    PRODUCT,  // 求积
    MAX,      // 最大值
    MIN,      // 最小值
};

class ThreadGroupCC {
   public:
    static ThreadGroupCC &GetInstance(size_t numThreads = 2);  // 2: thread group at least two thread

    ~ThreadGroupCC() = default;

    template <typename T>
    void AllGather(const std::vector<T> &sendData, std::vector<std::vector<T>> &recvData, size_t idx);

   protected:
    explicit ThreadGroupCC(size_t numThreads);

    template <typename T>
    void AllGatherSend_(const std::vector<T> &sendData, size_t idx);
    template <typename T>
    void AllGatherRecv_(std::vector<std::vector<T>> &recvData, size_t idx);

    template <typename T>
    static void CopyData2Buf_(const std::vector<T> &src, std::vector<std::any> &dst);
    template <typename T>
    static void CopyBuf2Data_(const std::vector<std::any> &src, std::vector<T> &dst);

   private:
    size_t numThreads_;  // shared by all cc primitives

    // barrier needed
    std::mutex barrierMtx_;
    size_t barrierCount_{0};
    size_t barrierPhase_{0};
    std::condition_variable barrierCv_;

    // broadcast needed
    std::mutex broadcastMtx_;
    std::condition_variable broadcastDataReadyCv_;
    std::condition_variable broadcastAllReadCv_;
    std::vector<std::any> broadcastBuf_;
    std::vector<bool> broadcastReadyVec_;
    size_t broadcastReadersDone_{0};

    // gather needed
    std::mutex gatherMtx_;
    std::condition_variable gatherDataReadyCv_;
    std::condition_variable gatherAllReadCv_;
    std::vector<std::vector<std::any>> gatherBuf_;
    std::vector<bool> gatherReadyVec_;
    std::vector<bool> gatherReaderDoneVec_;

    // scatter needed
    std::mutex scatterMtx_;
    std::condition_variable scatterDataReadyCv_;
    std::condition_variable scatterAllReadCv_;
    std::vector<std::vector<std::any>> scatterBuf_;
    std::vector<bool> scatterReadyVec_;
    size_t scatterReaderDone_{0};

    // allgather needed
    std::mutex allGatherMtx_;
    std::condition_variable allGatherDataReadyCv_;
    std::condition_variable allGatherAllReadCv_;
    std::vector<std::vector<std::any>> allGatherBuf_;
    std::vector<std::vector<bool>> allGatherReadyVec_;
    std::vector<std::vector<bool>> allGatherReaderDoneVec_;

    // reduce needed
    std::mutex reduceMtx_;
    std::condition_variable reduceDataReadyCv_;
    std::condition_variable reduceAllReadCv_;
    std::vector<std::vector<std::any>> reduceBuf_;
    std::vector<bool> reduceReadyVec_;
    std::vector<bool> reduceReaderDoneVec_;
};

template <typename T>
void ThreadGroupCC::AllGather(const std::vector<T> &sendData, std::vector<std::vector<T>> &recvData, size_t idx) {
    AllGatherSend_(sendData, idx);
    AllGatherRecv_(recvData, idx);
}

template <typename T>
void ThreadGroupCC::AllGatherSend_(const std::vector<T> &sendData, size_t idx) {
    if (idx >= allGatherBuf_.size()) {
        throw std::out_of_range("AllGather index out of range: " + std::to_string(idx) +
                                " >= " + std::to_string(allGatherBuf_.size()));
    }

    // 1. sender发送数据，每个线程将自己的数据放入缓冲区
    CopyData2Buf_(sendData, allGatherBuf_[idx]);
    std::unique_lock<std::mutex> lock(allGatherMtx_);
    for (size_t i = 0; i < numThreads_; ++i) {
        if (idx >= allGatherReadyVec_[i].size() || idx >= allGatherReaderDoneVec_[i].size()) {
            throw std::runtime_error("AllGather index out of range: " + std::to_string(idx) + " >= numThreads_" +
                                     std::to_string(i));
        }
        allGatherReadyVec_[i][idx] = true;
        allGatherReaderDoneVec_[i][idx] = false;
    }
    allGatherDataReadyCv_.notify_all();
}

template <typename T>
void ThreadGroupCC::AllGatherRecv_(std::vector<std::vector<T>> &recvData, size_t idx) {
    {
        std::unique_lock<std::mutex> lock(allGatherMtx_);

        // 1. 等待sender准备好数据
        allGatherDataReadyCv_.wait(lock, [this, idx] {
            return std::all_of(allGatherReadyVec_[idx].begin(), allGatherReadyVec_[idx].end(),
                               [](bool ready) { return ready; });
        });
    }

    // 2. reciever读取数据
    recvData.resize(numThreads_);
    for (size_t i = 0; i < numThreads_; ++i) {
        CopyBuf2Data_(allGatherBuf_[i], recvData[i]);
    }
    std::unique_lock<std::mutex> lock(allGatherMtx_);
    std::fill(allGatherReadyVec_[idx].begin(), allGatherReadyVec_[idx].end(), false);
    std::fill(allGatherReaderDoneVec_[idx].begin(), allGatherReaderDoneVec_[idx].end(), true);
    allGatherAllReadCv_.notify_all();

    // 3. 等待当前线程的元素都read完，就可以退出进入下一次集合通信
    allGatherAllReadCv_.wait(lock, [this, idx] {
        for (size_t i = 0; i < numThreads_; ++i) {
            if (!allGatherReaderDoneVec_[i][idx]) {
                return false;
            }
        }
        return true;
    });
}

template <typename T>
void ThreadGroupCC::CopyData2Buf_(const std::vector<T> &src, std::vector<std::any> &dst) {
    dst.resize(src.size());
    for (size_t i = 0; i < dst.size(); ++i) {
        dst[i] = std::make_any<T>(src[i]);
    }
}
template <typename T>
void ThreadGroupCC::CopyBuf2Data_(const std::vector<std::any> &src, std::vector<T> &dst) {
    dst.resize(src.size());
    for (size_t i = 0; i < dst.size(); ++i) {
        dst[i] = std::any_cast<T>(src[i]);
    }
}
}  // namespace mindie_llm

#endif
