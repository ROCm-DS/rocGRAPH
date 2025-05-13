/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_clients_expect_le.hpp"
#include "rocgraph_clients_are_le.hpp"
#include "rocgraph_clients_fail.hpp"

#include <iostream>

template <typename T>
bool rocgraph_clients_expect_le(const char* a_name,
                                const T     a,
                                const char* b_name,
                                const T     b,
                                const char* file_name,
                                int32_t     file_line)
{
    const bool are_le = rocgraph_clients_are_le(a, b);
    if(are_le == false)
    {
        std::cerr << "*** ERROR ROCGRAPH_CLIENTS_EXPECT_LE(a,b)" << std::endl;
        std::cerr << "*   file: { " << std::endl;
        std::cerr << "*     name: '" << file_name << "'," << std::endl;
        std::cerr << "*     line: '" << file_line << "'" << std::endl;
        std::cerr << "*   }," << std::endl;
        std::cerr << "*   a: { " << std::endl;
        std::cerr << "*     name: '" << a_name << "'," << std::endl;
        std::cerr << "*     value: '" << a << "'" << std::endl;
        std::cerr << "*   }," << std::endl;
        std::cerr << "*   b: { " << std::endl;
        std::cerr << "*     name: '" << b_name << "'," << std::endl;
        std::cerr << "*     value: '" << b << "'" << std::endl;
        std::cerr << "*   }" << std::endl;
        return false;
    }
    return true;
}

#define INSTANTIATE(T)                                              \
    template bool rocgraph_clients_expect_le(const char* a_name,    \
                                             const T     a,         \
                                             const char* b_name,    \
                                             const T     b,         \
                                             const char* file_name, \
                                             int32_t     file_line)

INSTANTIATE(size_t);
INSTANTIATE(int32_t);
INSTANTIATE(int64_t);
INSTANTIATE(float);
INSTANTIATE(double);

#undef INSTANTIATE
