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
#include "safe_envvar.h"

#include <Python.h>

#include <mutex>

namespace mindie_llm {

static std::string GetSitePackagesPath() {
    PyGILState_STATE gil = PyGILState_Ensure();
    std::string result;
    PyObject* site_module = PyImport_ImportModule("site");
    if (!site_module) {
        PyErr_Print();
        PyGILState_Release(gil);
        return "";
    }
    PyObject* func = PyObject_GetAttrString(site_module, "getsitepackages");
    if (func && PyCallable_Check(func)) {
        PyObject* list = PyObject_CallObject(func, nullptr);
        if (list && PyList_Size(list) > 0) {
            PyObject* item = PyList_GetItem(list, 0);
            if (PyUnicode_Check(item)) {
                result = PyUnicode_AsUTF8(item);
            }
        }
        Py_XDECREF(list);
    }
    Py_XDECREF(func);
    Py_DECREF(site_module);
    PyGILState_Release(gil);
    return result;
}

const std::string& GetDefaultMindIELLMHomePath() {
    static std::string path;
    static std::once_flag once;
    std::call_once(once, [] { path = GetSitePackagesPath() + "/mindie_llm/"; });
    return path;
}

EnvVar& EnvVar::GetInstance() {
    static EnvVar instance;
    return instance;
}

Result EnvVar::Set(const char* key, const std::string& value, bool overwrite) const {
    if (!key || value.empty()) {
        return Result::Error(ResultCode::NONE_ARGUMENT,
                             "Environment variable key is null or value is an empty string.");
    }
    int ret = setenv(key, value.c_str(), overwrite ? 1 : 0);
    if (ret != 0) {
        return Result::Error(ResultCode::IO_FAILURE, "Failed to set environment variable, errno: " +
                                                         std::to_string(errno) + " for key: " + std::string(key));
    }
    return Result::OK();
}

Result EnvVar::Get(const char* key, const std::string& defaultValue, std::string& outValue) const {
    if (!key || defaultValue.empty()) {
        return Result::Error(ResultCode::NONE_ARGUMENT,
                             "Environment variable key is nullptr or default value is empty.");
    }
    try {
        const char* val = std::getenv(key);
        outValue = (val) ? std::string(val) : defaultValue;
    } catch (const std::exception& e) {
        return Result::Error(ResultCode::IO_FAILURE, e.what());
    } catch (...) {
        return Result::Error(ResultCode::IO_FAILURE, "Unknown error occurred while fetching environment variable");
    }
    return Result::OK();
}

}  // namespace mindie_llm
