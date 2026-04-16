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

#include <iostream>
#include <string>

#include "post_processing_manager.h"
#include "post_processing_profiler/profiler.h"

namespace py = pybind11;

PYBIND11_MODULE(_cpu_logits_handler, m) {
    (void)py::class_<PostProcessingManager, std::shared_ptr<PostProcessingManager>>(m, "_PostProcessingManager")
        .def_static("get_instance", &PostProcessingManager::Instance, "Get AfterProcessManage instance")
        .def("next_token_chooser", &PostProcessingManager::NextTokenChooser, "next token chooser",
             py::arg("requestIds"), py::arg("scores_ptr"), py::arg("index_ptr"), py::arg("batchSize"),
             py::arg("scoreSize"), py::arg("maxLogprobs"), py::arg("dtype") = "float16", py::arg("speedMode") = false,
             py::arg("useApprox") = false)
        .def("set_batch_configs", &PostProcessingManager::SetBatchConfigs, "set batch configs", py::arg("requestIds"),
             py::arg("top_k"), py::arg("top_p"), py::arg("sample"), py::arg("logprobs"), py::arg("seed"),
             py::arg("sample_method") = "exponential")
        .def("delete_configs", &PostProcessingManager::DeleteConf, "delete conf");

    py::class_<PostProcessingProfiler::TimeCost>(m, "_TimeCost")
        .def(py::init<std::string, std::string, std::string>(), py::arg("name"), py::arg("pid"), py::arg("tid"))
        .def_readwrite("start", &PostProcessingProfiler::TimeCost::start)
        .def_readwrite("duration", &PostProcessingProfiler::TimeCost::duration)
        .def_readwrite("name", &PostProcessingProfiler::TimeCost::name)
        .def_readwrite("pid", &PostProcessingProfiler::TimeCost::pid)
        .def_readwrite("tid", &PostProcessingProfiler::TimeCost::tid);

    (void)py::class_<PostProcessingProfiler::Profiler, std::shared_ptr<PostProcessingProfiler::Profiler>>(m,
                                                                                                          "_Profiler")
        .def("get_instance", &PostProcessingProfiler::Profiler::GetInstance, "Get Profiler instance",
             py::return_value_policy::reference)
        .def("timer_start", &PostProcessingProfiler::Profiler::TimerStart, "start timer", py::arg("name"),
             py::arg("pid"), py::arg("tid"))
        .def("timer_end", &PostProcessingProfiler::Profiler::TimerEnd, "end timer")
        .def("export_result", &PostProcessingProfiler::Profiler::ExportResult, "export result",
             py::return_value_policy::reference)
        .def("set_activate", &PostProcessingProfiler::Profiler::SetActivate, "set activate");
}
