# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from setuptools import setup


setup(
    name="atb_llm",
    version="0.0.1",
    author="",
    author_email="",
    description="ATB LLM Project",
    long_description="",
    package_dir={"atb_llm": "atb_llm"},
    package_data={"": ["*.xlsx", "*.h5", "*.csv", "*.so", "*.avsc", "*.xml", "*.pkl", "*.sql", "*.ini", "*.json"]},
    zip_safe=False,
    python_requires=">=3.7",
)
