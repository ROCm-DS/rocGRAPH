/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_clients_expect_near_tolerance.hpp"
#include "rocgraph_clients_are_near_tolerance.hpp"
#include "rocgraph_clients_fail.hpp"

#include <iostream>

template <typename T>
bool rocgraph_clients_expect_near_tolerance(const char*        a_name,
                                            const T            a,
                                            const char*        b_name,
                                            const T            b,
                                            floating_data_t<T> tol,
                                            const char*        file_name,
                                            int32_t            file_line)
{
    const bool are_near = rocgraph_clients_are_near_tolerance(a, b, tol);
    if(are_near == false)
    {
        std::cerr << "*** ERROR ROCGRAPH_CLIENTS_EXPECT_NEAR_TOLERANCE(a, b, tol)" << std::endl;
        std::cerr << "*   file: { " << std::endl;
        std::cerr << "*     name:  '" << file_name << "'," << std::endl;
        std::cerr << "*     line:  '" << file_line << "'" << std::endl;
        std::cerr << "*   }," << std::endl;
        std::cerr << "*   a: { " << std::endl;
        std::cerr << "*     name:  '" << a_name << "'," << std::endl;
        std::cerr << "*     value: '" << a << "'" << std::endl;
        std::cerr << "*   }," << std::endl;
        std::cerr << "*   b: { " << std::endl;
        std::cerr << "*     name:  '" << b_name << "'," << std::endl;
        std::cerr << "*     value: '" << b << "'" << std::endl;
        std::cerr << "*   }" << std::endl;
        std::cerr << "*   tol: { " << std::endl;
        std::cerr << "*     value: '" << tol << "'" << std::endl;
        std::cerr << "*   }," << std::endl;
        const auto diff_val  = std::abs(a - b);
        const auto bound_val = std::max(std::abs(a), std::abs(b)) * tol;
        std::cerr << "*   diff: { value: '" << diff_val << "'," << std::endl;
        std::cerr << "*           bound: '" << bound_val << "' }" << std::endl;
        return false;
    }
    return true;
}

#define INSTANTIATE(T)                                    \
    template bool rocgraph_clients_expect_near_tolerance( \
        const char*, const T, const char*, const T, floating_data_t<T>, const char*, int32_t)

INSTANTIATE(float);
INSTANTIATE(double);

#undef INSTANTIATE
