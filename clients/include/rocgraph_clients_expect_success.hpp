/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph-types.h"
#include "rocgraph_clients_fail.hpp"

bool rocgraph_clients_expect_success(const char*     status_name,
                                     rocgraph_status status,
                                     const char*     file_name,
                                     int32_t         file_line);

//
// Check the rocgraph_status is success.
//
#define CHECK_ROCGRAPH_SUCCESS(STATUS_)                                                           \
    do                                                                                            \
    {                                                                                             \
        const bool eq = rocgraph_clients_expect_success(#STATUS_, (STATUS_), __FILE__, __LINE__); \
        if(eq == false)                                                                           \
        {                                                                                         \
            ROCGRAPH_CLIENTS_FAIL();                                                              \
        }                                                                                         \
    } while(false)
