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

#include <Python.h>

#include "pylog.h"

namespace FOUNDATION {
PyDoc_STRVAR(InterfaceModuleDoc, "The part of the MindIE-LLM module that is implemented in CXX.");

static PyModuleDef g_InterfaceModule = {
    PyModuleDef_HEAD_INIT,
    "foundation",        // m_name
    InterfaceModuleDoc,  // m_doc
    -1,                  // m_size
    nullptr,             // m_methods
    nullptr,             // m_slots
    nullptr,             // m_traverse
    nullptr,             // m_clear
    nullptr              // m_free
};

PyMODINIT_FUNC PyInit_foundation(void) {
    PyObject* m = PyModule_Create(&g_InterfaceModule);
    if (!m) {
        return nullptr;
    }
    PyObject* pyLog = GetLogModule();
    if (!pyLog) {
        PyErr_SetString(PyExc_ImportError, "Failed to create log submodule.");
        Py_DECREF(m);
        return nullptr;
    }
    Py_INCREF(pyLog);
    if (PyModule_AddObject(m, "log", pyLog) < 0) {
        PyErr_SetString(PyExc_ImportError, "Failed to add log submodule.");
        Py_DECREF(pyLog);
        Py_DECREF(m);
        return nullptr;
    }
    return m;
}
}  // namespace FOUNDATION
