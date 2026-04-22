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

#ifndef MINDIE_LLM_THREAD_POOL_H
#define MINDIE_LLM_THREAD_POOL_H

#include "task_queue.h"

namespace mindie_llm {
namespace cpu_logits_handler {
class ThreadPool {
   public:
    explicit ThreadPool(int threadNum) : m_threadNum(threadNum), m_taskNum(0) {};
    void Init();
    ~ThreadPool();

    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    void AddTask(Task task);
    void Join();

   private:
    static void *Worker(void *arg);
    void ThreadExit();

   private:
    pthread_mutex_t m_lock;
    pthread_mutex_t m_taskLock;
    pthread_cond_t m_notEmpty;
    pthread_cond_t m_empty;
    pthread_t *m_threadIDs = nullptr;
    TaskQueue *m_taskQ = nullptr;
    int m_threadNum;
    int m_taskNum;
    bool m_shutdown = false;
};
}  // namespace cpu_logits_handler
}  // namespace mindie_llm

#endif  // THREAD_POOL_H
