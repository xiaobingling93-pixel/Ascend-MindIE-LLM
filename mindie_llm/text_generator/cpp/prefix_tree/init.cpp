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

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "prefix_tree.h"

namespace py = pybind11;
PYBIND11_MODULE(_prefix_tree, m) {
    py::class_<mindie_llm::prefix_tree::PrefixTree>(m, "_PrefixTree")
        .def(py::init<int, int, int>())
        .def("put", &mindie_llm::prefix_tree::PrefixTree::Put)
        .def("get_one_draft", &mindie_llm::prefix_tree::PrefixTree::GetOneDraft)
        .def("reset_input_freq", &mindie_llm::prefix_tree::PrefixTree::ResetInputFreq, py::arg("batchId"))
        .def("trim", &mindie_llm::prefix_tree::PrefixTree::Trim);
}
