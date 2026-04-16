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

#ifndef LOCKED_DEQUE_H
#define LOCKED_DEQUE_H
#include <pthread.h>

#include <deque>

namespace mindie_llm {
template <typename T>
class ConcurrentDeque {
   public:
    ConcurrentDeque() { pthread_spin_init(&spinlock, PTHREAD_PROCESS_PRIVATE); }

    ~ConcurrentDeque() { pthread_spin_destroy(&spinlock); }

    void PushFront(const T &value) {
        pthread_spin_lock(&spinlock);
        deque.push_front(value);
        pthread_spin_unlock(&spinlock);
    }

    void PushBack(const T &value) {
        pthread_spin_lock(&spinlock);
        deque.push_back(value);
        pthread_spin_unlock(&spinlock);
    }

    bool PopFront(T &result) {
        pthread_spin_lock(&spinlock);
        if (deque.empty()) {
            pthread_spin_unlock(&spinlock);
            return false;
        }
        result = deque.front();
        deque.pop_front();
        pthread_spin_unlock(&spinlock);
        return true;
    }

    T Front() {
        pthread_spin_lock(&spinlock);
        if (deque.empty()) {
            pthread_spin_unlock(&spinlock);
            return nullptr;
        }
        T result = deque.front();
        pthread_spin_unlock(&spinlock);
        return result;
    }

    // 边云动态切块新增
    T Back() {
        pthread_spin_lock(&spinlock);
        if (deque.empty()) {
            pthread_spin_unlock(&spinlock);
            return nullptr;
        }
        T result = deque.back();
        pthread_spin_unlock(&spinlock);
        return result;
    }

    bool PopBack(T &result) {
        pthread_spin_lock(&spinlock);
        if (deque.empty()) {
            pthread_spin_unlock(&spinlock);
            return false;
        }
        result = deque.back();
        deque.pop_back();
        pthread_spin_unlock(&spinlock);
        return true;
    }

    bool Empty() {
        pthread_spin_lock(&spinlock);
        bool is_empty = deque.empty();
        pthread_spin_unlock(&spinlock);
        return is_empty;
    }

    bool Clear() {
        pthread_spin_lock(&spinlock);
        deque.clear();
        bool ret = deque.size() == 0;
        pthread_spin_unlock(&spinlock);
        return ret;
    }

    size_t Size() {
        pthread_spin_lock(&spinlock);
        size_t ret = deque.size();
        pthread_spin_unlock(&spinlock);
        return ret;
    }

    // 遍历并发队列，统计队列中请求的指标
    template <typename Func>
    void ForEach(Func func, size_t limit = 0) {
        if (Empty()) {
            return;
        }
        if (limit == 0) {
            return;
        }
        size_t count = 0;
        pthread_spin_lock(&spinlock);
        for (auto &it : deque) {
            if (count == limit) {
                break;
            }
            func(it);
            count++;
        }
        pthread_spin_unlock(&spinlock);
    }

   private:
    std::deque<T> deque;
    pthread_spinlock_t spinlock;
};
}  // namespace mindie_llm
#endif
