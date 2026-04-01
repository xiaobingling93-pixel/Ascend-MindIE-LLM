#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
import glob
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
USE_NINJA = os.getenv('USE_NINJA') == '1'
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if "--ops" in sys.argv:
    idx = sys.argv.index("--ops")
    if idx + 1 < len(sys.argv):
        ops = sys.argv[idx + 1]
        del sys.argv[idx:idx + 2] # 从 argv 中移除 --ops 及其值，避免 setuptools 报错
    else:
        raise ValueError("Missing the value of input parameter '--ops', mie_ops setup failed!")
else:
    raise ValueError("Missing input parameter '--ops', mie_ops setup failed!")
mie_ops_version = "mie_ops_" + ops

source_files = glob.glob(os.path.join(BASE_DIR, "mie_ops/torch_ops_extension", "*.cpp"), recursive=True)

exts = []
ext = NpuExtension(
    name=mie_ops_version + ".mie_ops_lib",
    sources=source_files,
    extra_compile_args=[
        '-I' + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include"),
        '-I' + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
        '-Wno-unused-variable',
        '-Wno-unused-parameter',
        '-Wno-unused-function',
        '-Wno-narrowing',
    ],
    extra_link_args=[
        '-L' + os.path.join(PYTORCH_NPU_INSTALL_PATH, "lib"),  # torch_npu 的 so 所在目录
        '-ltorch_npu',  # 链接 libtorch_npu.so
        '-ltorch',      # 可能还需要链接 libtorch.so
        '-lc10',        # PyTorch 依赖的库
        '-Wl,-rpath,/path/to/torch_npu/lib',  # 运行时 rpath
    ],
)
exts.append(ext)

setup(
    name=mie_ops_version,
    version='1.0',
    keywords=mie_ops_version,
    packages=[mie_ops_version],
    package_dir={mie_ops_version: "mie_ops"},  # 将实际版本的mie_ops映射到mie_ops目录
    ext_modules=exts,
    package_data={
        mie_ops_version: ['*.py', '*.so', 'opp/**/*'],
        mie_ops_version + '.torch_ops_extension': ['*.py', '*.so'],
    },
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
)
