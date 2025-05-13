/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph-types.h"
#include <hip/hip_runtime_api.h>

bool rocgraph_clients_expect_hip_success(const char* status_name,
                                         hipError_t  status,
                                         const char* file_name,
                                         int32_t     file_line);

//
// Check the hipError_t is success.
//
#define CHECK_HIP_SUCCESS(ERROR_)                                                         \
    do                                                                                    \
    {                                                                                     \
        const bool eq                                                                     \
            = rocgraph_clients_expect_hip_success(#ERROR_, (ERROR_), __FILE__, __LINE__); \
        if(eq == false)                                                                   \
        {                                                                                 \
            ROCGRAPH_CLIENTS_FAIL();                                                      \
        }                                                                                 \
    } while(false)
