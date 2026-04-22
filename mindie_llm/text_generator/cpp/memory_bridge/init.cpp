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

#include <pybind11/pybind11.h>  // 使用pybind11将函数和类封装为python包
#include <pybind11/stl.h>       // 函数中用到了C++的STL库，所以要包含该头文件

#include "swapper.h"

namespace hmm {
PYBIND11_MODULE(_memory_bridge, m) {
    m.doc() = "_memory_bridge: use memory bridge to manage heterogeneous memories";
    pybind11::class_<Swapper>(m, "_Swapper")
        .def_static("get_instance", &Swapper::Create, "Get Swapper instance")
        .def("h2d_swap", &Swapper::H2dSwap)
        .def("d2h_swap", &Swapper::D2hSwap);
}
}  // namespace hmm
