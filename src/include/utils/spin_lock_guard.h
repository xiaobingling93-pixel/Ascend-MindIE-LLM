/*
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

#ifndef SPIN_LOCK_GUARD
#define SPIN_LOCK_GUARD

#include <pthread.h>

namespace mindie_llm {
class SpinLockGuard {
   public:
    explicit SpinLockGuard(pthread_spinlock_t &lock) : spinlock_(lock) { pthread_spin_lock(&spinlock_); }
    ~SpinLockGuard() { pthread_spin_unlock(&spinlock_); }

   private:
    pthread_spinlock_t &spinlock_;
};
}  // namespace mindie_llm

#endif
