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

#ifndef MINDIE_LLM_TASK_QUEUE_H
#define MINDIE_LLM_TASK_QUEUE_H

#include <pthread.h>

#include <queue>

using Callback = void (*)(void *);

namespace mindie_llm {
namespace cpu_logits_handler {
struct Task {
    Task() : function(nullptr), arg(nullptr) {}
    Task(Callback f, void *arg) : function(f), arg(arg) {}
    Callback function;
    void *arg;
};

class TaskQueue {
   public:
    TaskQueue() = default;
    void Init();
    ~TaskQueue();

    void AddTask(const Task &task);
    void AddTask(Callback func, void *arg);

    Task TakeTask();

    inline size_t TaskNumber() { return m_queue.size(); }

   private:
    pthread_mutex_t m_mutex;
    std::queue<Task> m_queue;
};
}  // namespace cpu_logits_handler
}  // namespace mindie_llm

#endif  // TASK_QUEUE_H
