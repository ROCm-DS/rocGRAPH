/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph_clients_near_check.hpp"
#include "rocgraph_clients_unit_check.hpp"
#include "rocgraph_math.hpp"
#include "rocgraph_status_to_string.hpp"

#include <iostream>

#include "rocgraph_clients_expect_hip_success.hpp"
#include "rocgraph_clients_expect_status.hpp"
#include "rocgraph_clients_expect_success.hpp"
#include "rocgraph_clients_expect_success_error.hpp"

//
// Check the hipError_t is success.
//
#define CHECK_HIP_THROW_ERROR(ERROR)                   \
    do                                                 \
    {                                                  \
        auto CHECK_HIP_THROW_ERROR_status = (ERROR);   \
        if(CHECK_HIP_THROW_ERROR_status != hipSuccess) \
        {                                              \
            throw CHECK_HIP_THROW_ERROR_status;        \
        }                                              \
    } while(false)

//
// Check the rocgraph_status is success.
//
#define CHECK_ROCGRAPH_THROW_ERROR(STATUS)                               \
    do                                                                   \
    {                                                                    \
        auto CHECK_ROCGRAPH_THROW_ERROR_status = (STATUS);               \
        if(CHECK_ROCGRAPH_THROW_ERROR_status != rocgraph_status_success) \
        {                                                                \
            throw CHECK_ROCGRAPH_THROW_ERROR_status;                     \
        }                                                                \
    } while(false)
