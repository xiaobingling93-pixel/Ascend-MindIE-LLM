#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from enum import Enum


class ModelWrapperErrorCode(Enum):
    SUCCESS = 0
    PD_UNLINK_ERROR = 2000
    PD_LINK_ERROR = 2001
    PD_SWITCH_ROLE_ERROR = 2002
    PD_PULL_KV_ERROR = 2003
    PD_PULL_KV_UNKNOWN_ERROR = 2004
    PD_UNKNOWN_ERROR = 2005
    TRANSFER_KV_ERROR = 2006
    CLEAR_KV_ERROR = 2007
    PD_Link_QUERY_ERROR = 2008
