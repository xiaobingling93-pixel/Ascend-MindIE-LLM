/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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
#include <pybind11/stl_bind.h>

#include <atomic>

#include "common_util.h"
#include "error.h"
#include "llm_manager.h"
#include "memory_utils.h"
#include "status.h"

using namespace mindie_llm;
namespace py = pybind11;

namespace mindie_llm {
constexpr uint32_t MAX_INPUTS_NUM = 4 * 1024 * 1024;
constexpr uint32_t MAX_BYTE_ALLOWED = 4 * 1024 * 1024 * sizeof(int64_t);
void StructDefine(py::module &m) {
    py::enum_<MemType>(m, "MemType").value("HOST_MEM", MemType::HOST_MEM);

    py::enum_<InferDataType>(m, "InferDataType")
        .value("TYPE_INVALID", InferDataType::TYPE_INVALID)
        .value("TYPE_BOOL", InferDataType::TYPE_BOOL)
        .value("TYPE_UINT8", InferDataType::TYPE_UINT8)
        .value("TYPE_UINT16", InferDataType::TYPE_UINT16)
        .value("TYPE_UINT32", InferDataType::TYPE_UINT32)
        .value("TYPE_UINT64", InferDataType::TYPE_UINT64)
        .value("TYPE_INT8", InferDataType::TYPE_INT8)
        .value("TYPE_INT16", InferDataType::TYPE_INT16)
        .value("TYPE_INT32", InferDataType::TYPE_INT32)
        .value("TYPE_INT64", InferDataType::TYPE_INT64)
        .value("TYPE_FP16", InferDataType::TYPE_FP16)
        .value("TYPE_FP32", InferDataType::TYPE_FP32)
        .value("TYPE_FP64", InferDataType::TYPE_FP64)
        .value("TYPE_STRING", InferDataType::TYPE_STRING)
        .value("TYPE_BF16", InferDataType::TYPE_BF16)
        .value("TYPE_BUTT", InferDataType::TYPE_BUTT);

    py::enum_<Error::Code>(m, "Code")
        .value("OK", Error::Code::OK)
        .value("ERROR", Error::Code::ERROR)
        .value("INVALID_ARG", Error::Code::INVALID_ARG)
        .value("NOT_FOUND", Error::Code::NOT_FOUND);

    py::enum_<InferRequestId::DataType>(m, "DataType")
        .value("UINT64", InferRequestId::DataType::UINT64)
        .value("STRING", InferRequestId::DataType::STRING);

    py::enum_<StatusResponseType>(m, "StatusResponseType")
        .value("CONTROL_SIGNAL_STATUS", StatusResponseType::CONTROL_SIGNAL_STATUS)
        .value("REQUEST_ENQUEUE_STATUS", StatusResponseType::REQUEST_ENQUEUE_STATUS);

    py::enum_<Operation>(m, "Operation").value("STOP", Operation::STOP).value("RELEASE_KV", Operation::RELEASE_KV);
}

void StatusDefine(py::module &m) {
    py::class_<Status>(m, "Status")
        .def(py::init<Error::Code>())
        .def(py::init<Error::Code, const std::string>())
        .def(py::init<Error>())
        .def("is_ok", &Status::IsOk)
        .def("status_code", &Status::StatusCode)
        .def("status_msg", &Status::StatusMsg);
}

void ErrorDefine(py::module &m) {
    py::class_<Error>(m, "Error")
        .def(py::init<Error::Code>())
        .def(py::init<Error::Code, std::string>())
        .def("error_code", &Error::ErrorCode)
        .def("message", &Error::Message)
        .def("is_ok", &Error::IsOk);
}

py::dtype GetNumpyDtype(InferDataType dataType) {
    switch (dataType) {
        case InferDataType::TYPE_BOOL:
            return pybind11::dtype::of<bool>();
        case InferDataType::TYPE_UINT8:
            return pybind11::dtype::of<uint8_t>();
        case InferDataType::TYPE_UINT16:
            return pybind11::dtype::of<uint16_t>();
        case InferDataType::TYPE_UINT32:
            return pybind11::dtype::of<uint32_t>();
        case InferDataType::TYPE_UINT64:
            return pybind11::dtype::of<uint64_t>();
        case InferDataType::TYPE_INT8:
            return pybind11::dtype::of<int8_t>();
        case InferDataType::TYPE_INT16:
            return pybind11::dtype::of<int16_t>();
        case InferDataType::TYPE_INT32:
            return pybind11::dtype::of<int32_t>();
        case InferDataType::TYPE_INT64:
            return pybind11::dtype::of<int64_t>();
        case InferDataType::TYPE_FP32:
            return pybind11::dtype::of<float>();
        case InferDataType::TYPE_FP64:
            return pybind11::dtype::of<double>();
        default:
            throw std::runtime_error("Unsupported data type");
    }
}

py::array TensorToNumpy(const InferTensor &tensor) {
    auto shape = tensor.GetShape();
    void *data = tensor.GetData();

    pybind11::dtype dtype = GetNumpyDtype(tensor.GetDataType());
    return pybind11::array(dtype, shape, data);
}

void InferTensorDefine(py::module &m) {
    using TensorMap = std::unordered_map<std::string, std::shared_ptr<InferTensor>>;
    py::class_<InferTensor, std::shared_ptr<InferTensor>>(m, "InferTensor")
        .def(py::init<>())
        .def(py::init<std::string, InferDataType, std::vector<int64_t>>(), py::arg("name"), py::arg("data_type"),
             py::arg("data_shape"))
        .def("get_shape", &InferTensor::GetShape)
        .def("get_size", &InferTensor::GetSize)
        .def("get_data_type", &InferTensor::GetDataType)
        .def("get_mem_type", &InferTensor::GetMemType)
        .def("get_data", &InferTensor::GetData)
        .def("get_name", &InferTensor::GetName)
        .def("allocate", &InferTensor::Allocate, py::arg("size"))
        .def("set_buffer",
             [](InferTensor &self, py::buffer &buf, bool needRelease) {
                 auto bufferInfo = buf.request();
                 if (bufferInfo.size < 0 || bufferInfo.size > MAX_INPUTS_NUM ||
                     static_cast<uint64_t>(bufferInfo.itemsize) > sizeof(int64_t)) {
                     std::string message = "The number of items or item size in input buffer is error. ";
                     throw std::runtime_error(message + "the number of items must in the range of [0, " +
                                              std::to_string(MAX_INPUTS_NUM) + "]." +
                                              "the item size must in the range of (0, 8].");
                 }
                 auto bufferSize = bufferInfo.size * bufferInfo.itemsize;
                 if (bufferSize > MAX_BYTE_ALLOWED || bufferSize <= 0) {
                     std::string mallocSize = std::to_string(bufferSize);
                     throw std::runtime_error("valid byte allowed is (0, " + std::to_string(MAX_BYTE_ALLOWED) +
                                              "). try to allocate " + mallocSize);
                 }
                 void *data = malloc(bufferSize);
                 if (data == nullptr) {
                     throw std::runtime_error("malloc data failed.");
                 }
                 try {
                     if (memcpy_s(data, bufferSize, bufferInfo.ptr, bufferSize) != 0) {
                         throw std::runtime_error("Error occured in set_buffer memcpy_s.");
                     }
                     if (bufferInfo.ndim != 1) {
                         throw std::runtime_error("Buffer must be one-dimensional.");
                     }
                     self.SetBuffer(data, bufferSize, needRelease);
                 } catch (const std::exception &e) {
                     free(data);
                     throw e;
                 }
             })
        .def("set_release", &InferTensor::SetRelease, py::arg("release_flag"))
        .def("release", &InferTensor::Release);

    m.def("tensor_to_numpy", &TensorToNumpy, "Converts the InferTensor's data to a NumPy array.");

    py::bind_map<TensorMap>(m, "TensorMap");
}

void InferRequestDefine(py::module &m) {
    py::class_<InferRequestId>(m, "InferRequestId")
        .def(py::init<>())
        .def(py::init<std::string>())
        .def(py::init<uint64_t>())
        .def("type", &InferRequestId::Type)
        .def("string_value", &InferRequestId::StringValue)
        .def("unsigned_int_value", &InferRequestId::UnsignedIntValue);
    py::class_<InferRequest, std::shared_ptr<InferRequest>>(m, "InferRequest")
        .def(py::init<InferRequestId>())
        .def("add_tensor", &InferRequest::AddTensor, py::arg("tensor_name"), py::arg("tensor"))
        .def("set_tensor", &InferRequest::SetTensor, py::arg("tensor_name"), py::arg("tensor"))
        .def("get_tensor_by_name", &InferRequest::GetTensorByName, py::arg("tensor_name"), py::arg("tensor"))
        .def("del_tensor_by_name", &InferRequest::DelTensorByName, py::arg("name"))
        .def("get_request_id", &InferRequest::GetRequestId)
        .def("set_max_output_len", &InferRequest::SetMaxOutputLen, py::arg("max_output_len"))
        .def("get_max_output_len", &InferRequest::GetMaxOutputLen)
        .def("immutable_inputs", &InferRequest::ImmutableInputs);
}

void LlmManagerDefine(py::module &m) {
    py::class_<LlmManager, std::shared_ptr<LlmManager>>(m, "LlmManager")
        .def(py::init<const std::string &, GetRequestsCallback, SendResponsesCallback, ControlSignalCallback,
                      LlmManagerStatsCallback, SendStatusResponseCallback>(),
             py::arg("llm_config_path"), py::arg("get_request"), py::arg("send_response"), py::arg("control_callback"),
             py::arg("status_callback"), py::arg("status_response_callback"))
        .def("get_max_position_embeddings", &LlmManager::GetMaxPositionEmbeddings)
        .def("shutdown", &LlmManager::Shutdown)
        .def("init", py::overload_cast<uint32_t, std::set<size_t>>(&LlmManager::Init), py::arg("model_instanceId"),
             py::arg("npu_device_ids"))
        .def("init",
             py::overload_cast<uint32_t, std::set<size_t>, std::map<std::string, std::string>>(&LlmManager::Init),
             py::arg("model_instanceId"), py::arg("npu_device_ids"), py::arg("extend_info"));
}

PYBIND11_MODULE(llm_manager_python, m) {
    StatusDefine(m);
    ErrorDefine(m);
    StructDefine(m);
    InferTensorDefine(m);
    InferRequestDefine(m);
    LlmManagerDefine(m);
}
}  // namespace mindie_llm
