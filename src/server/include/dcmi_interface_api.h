/*
 * Copyright: Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Author: huawei
 * Date: 2021-03-17 17:46:08
 * @LastEditors: huawei
 * @LastEditTime: 2022-11-03 11:17:04
 * Description: DCMI API Reference
 */

/***************************************************************************************/
#ifndef DCMI_INTERFACE_API_H
#define DCMI_INTERFACE_API_H

#ifdef __linux
#define DCMIDLLEXPORT
#else
#define DCMIDLLEXPORT _declspec(dllexport)
#endif

#define TOPO_INFO_MAX_LENTH 32  // topo info max length

DCMIDLLEXPORT int dcmi_init(void);

DCMIDLLEXPORT int dcmi_get_device_utilization_rate(int card_id, int device_id, int input_type,
                                                   unsigned int *utilization_rate);

#endif  // DCMI_INTERFACE_API_H
