/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_clients_expect_status.hpp"
#include "rocgraph_clients_fail.hpp"
#include "rocgraph_status_to_string.hpp"
#include <iostream>

bool rocgraph_clients_expect_status(const char*     status_expr,
                                    rocgraph_status status,
                                    const char*     expected_status_expr,
                                    rocgraph_status expected_status,
                                    const char*     file_name,
                                    int32_t         file_line)
{
    if(status != expected_status)
    {
        std::cerr << "*** ERROR ROCGRAPH_CLIENTS_EXPECT_STATUS(status)" << std::endl;
        std::cerr << "*   file: { " << std::endl;
        std::cerr << "*     name:  '" << file_name << "'," << std::endl;
        std::cerr << "*     line:  '" << file_line << "'" << std::endl;
        std::cerr << "*   }," << std::endl;
        std::cerr << "*   status: { " << std::endl;
        std::cerr << "*     expr:  '" << status_expr << "'," << std::endl;
        std::cerr << "*     name:  '" << rocgraph_status_to_string(status) << "'," << std::endl;
        std::cerr << "*     value: '" << status << "'" << std::endl;
        std::cerr << "*   }," << std::endl;
        std::cerr << "*   expected: { " << std::endl;
        std::cerr << "*     expr:  '" << expected_status_expr << "'," << std::endl;
        std::cerr << "*     name:  '" << rocgraph_status_to_string(expected_status) << "',"
                  << std::endl;
        std::cerr << "*     value: '" << expected_status << "'" << std::endl;
        std::cerr << "*   }" << std::endl;
        return false;
    }
    return true;
}
