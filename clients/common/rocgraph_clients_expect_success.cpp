/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_clients_expect_success.hpp"
#include "rocgraph_clients_fail.hpp"
#include "rocgraph_status_to_string.hpp"
#include <iostream>

bool rocgraph_clients_expect_success(const char*     status_expr,
                                     rocgraph_status status,
                                     const char*     file_name,
                                     int32_t         file_line)
{
    if(status != rocgraph_status_success)
    {
        std::cerr << "*** ERROR ROCGRAPH_CLIENTS_EXPECT_SUCCESS(status)" << std::endl;
        std::cerr << "*   file: { " << std::endl;
        std::cerr << "*     name:  '" << file_name << "'," << std::endl;
        std::cerr << "*     line:  '" << file_line << "'" << std::endl;
        std::cerr << "*   }," << std::endl;
        std::cerr << "*   status: { " << std::endl;
        std::cerr << "*     expr:  '" << status_expr << "'," << std::endl;
        std::cerr << "*     name:  '" << rocgraph_status_to_string(status) << "'," << std::endl;
        std::cerr << "*     value: '" << status << "'" << std::endl;
        std::cerr << "*   }" << std::endl;
        return false;
    }
    return true;
}
