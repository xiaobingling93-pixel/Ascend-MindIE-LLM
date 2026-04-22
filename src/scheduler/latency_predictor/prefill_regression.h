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

#ifndef PREFILL_REGRESSION
#define PREFILL_REGRESSION

namespace mindie_llm {
class PrefillRegression {
   public:
    PrefillRegression();
    ~PrefillRegression() = default;
    void AddDataPoint(float tokenNum, float execTime);
    float Predict(int tokenNum) const;

   private:
    int count_;
    float sumX_, sumY_, sumXX_, sumXY_;
    float slope_ = 0.1317;
    float intercept_ = 25.797;
    void LinearRegression(float tokenNum, float execTime);
};
}  // namespace mindie_llm

#endif  // PREFILL_REGRESSION
