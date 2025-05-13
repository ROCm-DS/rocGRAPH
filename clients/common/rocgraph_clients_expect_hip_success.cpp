/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_clients_expect_hip_success.hpp"
#include "rocgraph_clients_fail.hpp"

#include <iostream>

bool rocgraph_clients_expect_hip_success(const char* status_expr,
                                         hipError_t  status,
                                         const char* file_name,
                                         int32_t     file_line)
{
    if(status != hipSuccess)
    {
        std::cerr << "*** ERROR ROCGRAPH_CLIENTS_EXPECT_HIP_SUCCESS(status)" << std::endl;
        std::cerr << "*   file: { " << std::endl;
        std::cerr << "*     name:  '" << file_name << "'," << std::endl;
        std::cerr << "*     line:  '" << file_line << "'" << std::endl;
        std::cerr << "*   }," << std::endl;
        std::cerr << "*   status: { " << std::endl;
        std::cerr << "*     expr:  '" << status_expr << "'," << std::endl;
        std::cerr << "*     name:  '" << hipGetErrorString(status) << "'," << std::endl;
        std::cerr << "*     value: '" << status << "'" << std::endl;
        std::cerr << "*   }" << std::endl;
        return false;
    }
    return true;
}
