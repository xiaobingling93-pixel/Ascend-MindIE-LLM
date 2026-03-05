/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "pylog.h"

#include <Python.h>
#include <frameobject.h>

#include "system_log.h"

using namespace mindie_llm;

namespace FOUNDATION {
    PyDoc_STRVAR(LogModuleDoc, "C++ logging bindings based on system_log.");

    static void LogLineForPyObject(LogComponent comp, LogSeverity level, LogType type, const char* msg)
    {
        auto& mgr = LogManager::GetInstance();
        if (!mgr.IsPrintLog(comp, level)) {
            return;
        }

        const char* filename = "";
        size_t lineNum = 0;
        PyGILState_STATE gil = PyGILState_Ensure();
        PyFrameObject* frame = PyEval_GetFrame();
        PyFrameObject* caller = frame ? PyFrame_GetBack(frame) : nullptr;
        PyCodeObject* code = caller ? PyFrame_GetCode(caller) : nullptr;
        if (code) {
            PyObject* pyFile = code->co_filename;
            if (pyFile) filename = PyUnicode_AsUTF8(pyFile);
            lineNum = static_cast<size_t>(PyFrame_GetLineNumber(caller));
        }
        Py_XDECREF(code);
        Py_XDECREF(caller);
        PyGILState_Release(gil);

        LogLine line(comp, level, filename, lineNum);
        line.SetType(type) << msg;
    }

    static PyObject* PyLog(PyObject*, PyObject* args, LogSeverity level)
    {
        const char* msg;
        const char* compStr;
        const char* typeStr = "general";

        if (!PyArg_ParseTuple(args, "ss|s", &msg, &compStr, &typeStr)) {
            return nullptr;
        }
        LogComponent comp;
        if (!String2Component(compStr, comp)) {
            PyErr_Format(PyExc_ValueError, "invalid component: '%s'", compStr);
            return nullptr;
        }
        LogType type;
        if (!String2LogType(typeStr, type)) {
            PyErr_Format(PyExc_ValueError, "invalid log type: '%s'", typeStr);
            return nullptr;
        }
        LogLineForPyObject(comp, level, type, msg);
        Py_RETURN_NONE;
    }

    static PyObject* PyDebug(PyObject* s, PyObject* a)
    {
        return PyLog(s, a, LogSeverity::DEBUG);
    }

    static PyObject* PyInfo(PyObject* s, PyObject* a)
    {
        return PyLog(s, a, LogSeverity::INFO);
    }

    static PyObject* PyWarn(PyObject* s, PyObject* a)
    {
        return PyLog(s, a, LogSeverity::WARN);
    }

    static PyObject* PyError(PyObject* s, PyObject* a)
    {
        return PyLog(s, a, LogSeverity::ERROR);
    }

    static PyObject* PyCritical(PyObject* s, PyObject* a)
    {
        return PyLog(s, a, LogSeverity::CRITICAL);
    }

    static PyObject* PyAudit(PyObject* s, PyObject* a)
    {
        return PyLog(s, a, LogSeverity::AUDIT);
    }

    static PyObject* PySetLogLevel(PyObject*, PyObject* arg)
    {
        if (!PyUnicode_Check(arg)) {
            PyErr_SetString(PyExc_TypeError, "set_log_level(level: str), e.g. 'debug', 'llm:info;llmmodels:warn'"
            );
            return nullptr;
        }
        const char* levelStr = PyUnicode_AsUTF8(arg);
        if (levelStr == nullptr) {
            return nullptr;
        }
        const std::string level(levelStr);
        LogManager::GetInstance().LoadByComponentByString<LogSeverity>(level, GetAllLogSeverity(),
            [](const std::string& s) {
                LogSeverity lvl;
                if (!String2LogSeverity(s, lvl)) {
                    return LogSeverity::INFO;
                }
                return lvl;
            },
            [](ComponentConfig& c, LogSeverity v) {
                c.minLevel = v;
            }
        );
        Py_RETURN_NONE;
    }

    static PyMethodDef LogMethods[] = {
        {"debug",         PyDebug,       METH_VARARGS, "debug(msg, comp, 'general')"},
        {"info",          PyInfo,        METH_VARARGS, "info(msg, comp, 'general')"},
        {"warn",          PyWarn,        METH_VARARGS, "warn(msg, comp, 'general')"},
        {"error",         PyError,       METH_VARARGS, "error(msg, comp, 'general')"},
        {"critical",      PyCritical,    METH_VARARGS, "critical(msg, comp, 'general')"},
        {"audit",         PyAudit,       METH_VARARGS, "audit(msg, comp, 'general')"},
        {"set_log_level", PySetLogLevel, METH_O,       "set_log_level(level: str)"},
        {nullptr,         nullptr,       0,            nullptr}
    };

    static PyModuleDef g_LogModule = {
        PyModuleDef_HEAD_INIT,
        "foundation.log",   // m_name
        LogModuleDoc,       // m_doc
        -1,                 // m_size
        LogMethods,         // m_methods
        nullptr,            // m_slots
        nullptr,            // m_traverse
        nullptr,            // m_clear
        nullptr             // m_free
    };

    PyObject* GetLogModule()
    {
        return PyModule_Create(&g_LogModule);
    }

} // namespace FOUNDATION
