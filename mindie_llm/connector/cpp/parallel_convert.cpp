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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>  // 使用pybind11将函数和类封装为python包
#include <pybind11/stl.h>       // 函数中用到了C++的STL库，所以要包含该头文件

#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <cassert>
#include <future>
#include <iostream>
#include <vector>

#include "log.h"
#include "model_execute_data.pb.h"

namespace py = pybind11;

using model_execute_data::CompletionSequenceGroupOutput;
using model_execute_data::ExecuteModelResponse;
using model_execute_data::ExecuteRequest;
using model_execute_data::ExecuteResponse;
using model_execute_data::ExecuteType;
using model_execute_data::MODEL_INFER;
using model_execute_data::SequenceOutput;

namespace mindie_llm {

// 封装所有输入参数的结构体
struct ChunkProcessParams {
    const int64_t *group_indices_data;   // group索引数据
    const int32_t *num_top_tokens;       // top token数量数组
    const int64_t *sequence_ids;         // 序列ID数组
    const int64_t *parent_sequence_ids;  // 父序列ID数组
    const int64_t *eos_info;             // EOS信息数组
    const int64_t *truncation_indices;   // 截断索引数组
    int64_t num_parallel_tokens;         // 并行token数量
    const float *cumulative_logprobs;    // 累积对数概率数组
    const int64_t *token_ids;            // token ID数组
    const float *logprobs;               // 对数概率数组
    ssize_t top_token_ids_shape1;        // top_token_ids第一维度大小
    ssize_t top_token_ids_shape2;        // top_token_ids第二维度大小
    const int64_t *top_token_ids;        // top token ID数组
    const float *top_logprobs;           // top token对数概率数组
};

// 线程数为 CPU 核心数
static boost::asio::thread_pool thread_pool(boost::thread::hardware_concurrency());

void process_chunk(int index, const ChunkProcessParams &params, CompletionSequenceGroupOutput *group_output) {
    int64_t start_index = params.group_indices_data[index * 2];
    int64_t end_index = params.group_indices_data[index * 2 + 1];
    int64_t seq_count = end_index - start_index;
    // top token数量
    int64_t num_top_token = params.num_top_tokens[start_index];

    if (seq_count <= 0) {
        return;
    }

    // 处理每个序列
    for (int64_t i = 0; i < seq_count; ++i) {
        SequenceOutput *seq_output = group_output->add_samples();
        int current_idx = start_index + i;
        // 设置基本属性
        seq_output->set_seq_id(params.sequence_ids[current_idx]);
        seq_output->set_parent_seq_id(params.parent_sequence_ids[current_idx]);
        seq_output->set_num_speculative_tokens(params.eos_info[2 * current_idx + 1]);
        seq_output->set_finish_reason(static_cast<int32_t>(params.eos_info[2 * current_idx]));
        seq_output->set_truncation_index(params.truncation_indices[current_idx]);
        seq_output->set_num_parallel_tokens(params.num_parallel_tokens);
        seq_output->set_cumulative_logprobs(params.cumulative_logprobs[current_idx]);

        for (ssize_t j = 0; j < params.num_parallel_tokens; ++j) {
            seq_output->add_output_token(params.token_ids[current_idx * params.num_parallel_tokens + j]);
        }

        for (ssize_t j = 0; j < params.num_parallel_tokens; ++j) {
            seq_output->add_logprob(params.logprobs[current_idx * params.num_parallel_tokens + j]);
        }

        for (ssize_t j = 0; j < params.top_token_ids_shape1; ++j) {
            // 第二维的数据都要
            for (ssize_t k = 0; k < num_top_token; ++k) {
                // 第三维的数据只要前num_top_token个
                seq_output->add_top_token_ids(
                    params.top_token_ids[current_idx * params.top_token_ids_shape1 * params.top_token_ids_shape2 +
                                         j * params.top_token_ids_shape2 + k]);
            }
        }

        for (ssize_t j = 0; j < params.top_token_ids_shape1; ++j) {
            for (ssize_t k = 0; k < num_top_token; ++k) {
                seq_output->add_top_logprobs(
                    params.top_logprobs[current_idx * params.top_token_ids_shape1 * params.top_token_ids_shape2 +
                                        j * params.top_token_ids_shape2 + k]);
            }
        }
    }
}

// 主转换函数 - 使用Python中定义的Protobuf类
py::object convert_generate_output(const py::object &generate_output) {
    // 构造response
    ExecuteResponse cpp_response;
    cpp_response.set_status(0);
    // 只支持MODEL_INFER
    cpp_response.set_msg_type(1);
    auto *cpp_execute_model_response = cpp_response.mutable_execute_model_response();

    // 提取group_indices
    py::array group_indices_arr = generate_output.attr("group_indices").cast<py::array>();
    auto group_indices_numpy = group_indices_arr.request();
    int64_t *group_indices_data = static_cast<int64_t *>(group_indices_numpy.ptr);
    size_t num_groups = group_indices_numpy.size / 2;

    // 提取所有需要的字段
    auto sequence_ids = static_cast<int64_t *>(generate_output.attr("sequence_ids").cast<py::array>().request().ptr);
    auto parent_sequence_ids =
        static_cast<int64_t *>(generate_output.attr("parent_sequence_ids").cast<py::array>().request().ptr);
    auto num_top_tokens =
        static_cast<int32_t *>(generate_output.attr("num_top_tokens").cast<py::array>().request().ptr);

    // 二维数组
    auto token_ids_numpy = generate_output.attr("token_ids").cast<py::array>().request();
    int64_t *token_ids = static_cast<int64_t *>(token_ids_numpy.ptr);
    ssize_t num_parallel_tokens = token_ids_numpy.shape[1];

    // 二维数组
    auto logprobs_numpy = generate_output.attr("logprobs").cast<py::array>().request();
    float *logprobs = static_cast<float *>(logprobs_numpy.ptr);

    // 二维数组
    int64_t *eos_info = static_cast<int64_t *>(generate_output.attr("eos_info").cast<py::array>().request().ptr);
    int64_t *truncation_indices =
        static_cast<int64_t *>(generate_output.attr("truncation_indices").cast<py::array>().request().ptr);
    float *cumulative_logprobs =
        static_cast<float *>(generate_output.attr("cumulative_logprobs").cast<py::array>().request().ptr);
    // 三维数组
    auto top_token_ids_numpy = generate_output.attr("top_token_ids").cast<py::array>().request();
    int64_t *top_token_ids = static_cast<int64_t *>(top_token_ids_numpy.ptr);
    ssize_t top_token_ids_shape1 = top_token_ids_numpy.shape[1];
    ssize_t top_token_ids_shape2 = top_token_ids_numpy.shape[2];
    // 三维数组
    float *top_logprobs = static_cast<float *>(generate_output.attr("top_logprobs").cast<py::array>().request().ptr);

    ChunkProcessParams params{.group_indices_data = group_indices_data,
                              .num_top_tokens = num_top_tokens,
                              .sequence_ids = sequence_ids,
                              .parent_sequence_ids = parent_sequence_ids,
                              .eos_info = eos_info,
                              .truncation_indices = truncation_indices,
                              .num_parallel_tokens = num_parallel_tokens,
                              .cumulative_logprobs = cumulative_logprobs,
                              .token_ids = token_ids,
                              .logprobs = logprobs,
                              .top_token_ids_shape1 = top_token_ids_shape1,
                              .top_token_ids_shape2 = top_token_ids_shape2,
                              .top_token_ids = top_token_ids,
                              .top_logprobs = top_logprobs};

    std::vector<std::future<void>> futures;

    // 处理每个group_output
    for (size_t g = 0; g < num_groups; ++g) {
        CompletionSequenceGroupOutput *group_output = cpp_execute_model_response->add_outputs();

        // 使用shared_ptr管理packaged_task的生命周期，避免提前释放
        auto task_ptr = std::make_shared<std::packaged_task<void()>>(
            [g, &params, group_output]() {  // 注意：params按值捕获，避免悬空引用
                process_chunk(g, params, group_output);
            });

        // 获取future并保存
        futures.push_back(task_ptr->get_future());

        // 提交任务到线程池：捕获shared_ptr，确保任务执行期间packaged_task始终有效
        boost::asio::post(thread_pool, [task_ptr]() {
            (*task_ptr)();  // 执行任务
        });
    }
    // 等待所有任务完成
    for (auto &fut : futures) {
        fut.get();  // 阻塞直到当前任务完成，无返回值
    }

    const size_t msg_size = cpp_response.ByteSizeLong();
    std::string buffer;
    buffer.resize(msg_size);
    cpp_response.SerializeToArray(buffer.data(), static_cast<int>(msg_size));
    return pybind11::bytes(buffer);
}

// 边云特性使用
void lwd_build_cpp_response(ExecuteResponse &cpp_response, ChunkProcessParams &params, bool isPrefill,
                            size_t numGroups) {
    cpp_response.set_status(0);
    // 只支持MODEL_INFER
    cpp_response.set_msg_type(1);
    auto *cpp_execute_model_response = cpp_response.mutable_execute_model_response();
    // 返回batch类型
    cpp_execute_model_response->set_layerwise_is_prefill(isPrefill);
    std::vector<std::future<void>> futures;

    // 处理每个group_output
    for (size_t g = 0; g < numGroups; ++g) {
        CompletionSequenceGroupOutput *group_output = cpp_execute_model_response->add_outputs();

        // 使用shared_ptr管理packaged_task的生命周期，避免提前释放
        auto task_ptr = std::make_shared<std::packaged_task<void()>>(
            [g, &params, group_output]() {  // 注意：params按值捕获，避免悬空引用
                process_chunk(g, params, group_output);
            });

        // 获取future并保存
        futures.push_back(task_ptr->get_future());

        // 提交任务到线程池：捕获shared_ptr，确保任务执行期间packaged_task始终有效
        boost::asio::post(thread_pool, [task_ptr]() {
            (*task_ptr)();  // 执行任务
        });
    }
    // 等待所有任务完成
    for (auto &fut : futures) {
        fut.get();  // 阻塞直到当前任务完成，无返回值
    }
    return;
}
py::object lwd_convert_generate_output(const py::object &generate_output, bool is_prefill) {
    // 构造response
    ExecuteResponse cpp_response;

    // 提取group_indices
    py::array group_indices_arr = generate_output.attr("group_indices").cast<py::array>();
    auto group_indices_numpy = group_indices_arr.request();
    int64_t *groupIndicesData = static_cast<int64_t *>(group_indices_numpy.ptr);
    size_t numGroups = group_indices_numpy.size / 2;

    // 提取所有需要的字段
    auto sequence_ids = static_cast<int64_t *>(generate_output.attr("sequence_ids").cast<py::array>().request().ptr);
    auto parent_sequence_ids =
        static_cast<int64_t *>(generate_output.attr("parent_sequence_ids").cast<py::array>().request().ptr);
    auto num_top_tokens =
        static_cast<int32_t *>(generate_output.attr("num_top_tokens").cast<py::array>().request().ptr);

    // 二维数组
    auto token_ids_numpy = generate_output.attr("token_ids").cast<py::array>().request();
    int64_t *tokenIds = static_cast<int64_t *>(token_ids_numpy.ptr);
    ssize_t numParallelTokens = token_ids_numpy.shape[1];

    // 二维数组
    auto logprobs_numpy = generate_output.attr("logprobs").cast<py::array>().request();
    float *logprobs = static_cast<float *>(logprobs_numpy.ptr);

    // 二维数组
    int64_t *eosInfo = static_cast<int64_t *>(generate_output.attr("eos_info").cast<py::array>().request().ptr);
    int64_t *truncationIndices =
        static_cast<int64_t *>(generate_output.attr("truncation_indices").cast<py::array>().request().ptr);
    float *cumulativeLogprobs =
        static_cast<float *>(generate_output.attr("cumulative_logprobs").cast<py::array>().request().ptr);
    // 三维数组
    auto top_token_ids_numpy = generate_output.attr("top_token_ids").cast<py::array>().request();
    int64_t *topTokenIds = static_cast<int64_t *>(top_token_ids_numpy.ptr);
    ssize_t topTokenIdsShape1 = top_token_ids_numpy.shape[1];
    ssize_t topTokenIdsShape2 = top_token_ids_numpy.shape[2];
    // 三维数组
    float *topLogprobs = static_cast<float *>(generate_output.attr("top_logprobs").cast<py::array>().request().ptr);

    ChunkProcessParams params{.group_indices_data = groupIndicesData,
                              .num_top_tokens = num_top_tokens,
                              .sequence_ids = sequence_ids,
                              .parent_sequence_ids = parent_sequence_ids,
                              .eos_info = eosInfo,
                              .truncation_indices = truncationIndices,
                              .num_parallel_tokens = numParallelTokens,
                              .cumulative_logprobs = cumulativeLogprobs,
                              .token_ids = tokenIds,
                              .logprobs = logprobs,
                              .top_token_ids_shape1 = topTokenIdsShape1,
                              .top_token_ids_shape2 = topTokenIdsShape2,
                              .top_token_ids = topTokenIds,
                              .top_logprobs = topLogprobs};

    lwd_build_cpp_response(cpp_response, params, is_prefill, numGroups);
    const size_t msgSize = cpp_response.ByteSizeLong();
    std::string buffer;
    buffer.resize(msgSize);
    cpp_response.SerializeToArray(buffer.data(), static_cast<int>(msgSize));
    return pybind11::bytes(buffer);
}

PYBIND11_MODULE(_mindie_llm_connector, m) {
    m.doc() = "_mindie_llm_connector: C++ Methods Used by the Connector Module.";
    // 绑定转换函数
    m.def("convert_generate_output", &convert_generate_output, "Convert generate output to protobuf response",
          py::arg("generate_output"));
    m.def(
        "lwd_convert_generate_output",
        [](const py::object &generate_output, bool is_prefill = false) {
            return lwd_convert_generate_output(generate_output, is_prefill);
        },
        "Layerwise-Convert generate output to protobuf response", py::arg("generate_output"),
        py::arg("is_prefill") = false);
}
}  // namespace mindie_llm
