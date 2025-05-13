/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_clients_expect_array_near_tolerance.hpp"
#include "rocgraph_clients_are_near_tolerance.hpp"
#include "rocgraph_clients_fail.hpp"
#include <iostream>

template <typename T>
bool rocgraph_clients_expect_array_near_tolerance(size_t      size,
                                                  const char* a_name,
                                                  const T* __restrict__ a,
                                                  size_t      a_inc,
                                                  const char* b_name,
                                                  const T* __restrict__ b,
                                                  size_t             b_inc,
                                                  floating_data_t<T> tol,
                                                  const char*        file_name,
                                                  int32_t            file_line)
{
    static constexpr size_t s_max_count = 10;
    size_t                  indices[s_max_count];
    size_t                  count = 0;
    for(size_t i = 0; i < size; ++i)
    {
        const bool are_near = rocgraph_clients_are_near_tolerance(a[i * a_inc], b[i * b_inc], tol);
        if(are_near == false)
        {
            if(count < s_max_count)
            {
                indices[count] = i;
            }
            ++count;
        }
    }

    if(count > 0)
    {
        std::cerr << "*** ERROR ROCGRAPH_CLIENTS_EXPECT_ARRAY_NEAR_TOLERANCE(size, a, a_inc, b, "
                     "b_inc, tol)"
                  << std::endl;
        std::cerr << "*   file: { " << std::endl;
        std::cerr << "*     name:  '" << file_name << "'," << std::endl;
        std::cerr << "*     line:  '" << file_line << "'" << std::endl;
        std::cerr << "*   }," << std::endl;
        std::cerr << "*   a:    { " << std::endl;
        std::cerr << "*     name:  '" << a_name << "'," << std::endl;
        std::cerr << "*     inc:   '" << a_inc << "'," << std::endl;
        std::cerr << "*   }," << std::endl;
        std::cerr << "*   b:    { " << std::endl;
        std::cerr << "*     name:  '" << b_name << "'," << std::endl;
        std::cerr << "*     inc:   '" << b_inc << "'," << std::endl;
        std::cerr << "*   }," << std::endl;
        std::cerr << "*   tol:  { " << std::endl;
        std::cerr << "*     value: '" << tol << "'" << std::endl;
        std::cerr << "*   }," << std::endl;
        std::cerr << "*   values: { " << std::endl;
        for(size_t i = 0; i < std::min(s_max_count, count); ++i)
        {
            const size_t index     = indices[i];
            const auto   a_val     = a[index * a_inc];
            const auto   b_val     = b[index * b_inc];
            const auto   diff_val  = std::abs(a_val - b_val);
            const auto   bound_val = std::max(std::abs(a_val), std::abs(b_val)) * tol;
            if(i > 0)
            {
                std::cerr << "," << std::endl;
            }
            std::cerr << "*     { index: '" << index << "'," << std::endl;
            std::cerr << "*       a[" << index << "]: '" << a_val << "'," << std::endl;
            std::cerr << "*       b[" << index << "]: '" << b_val << "'," << std::endl;
            std::cerr << "*       diff:  '" << diff_val << "'," << std::endl;
            std::cerr << "*       bound: '" << bound_val << "' }";
        }
        if(count > s_max_count)
        {
            std::cerr << std::endl
                      << "*            ... truncated with " << s_max_count << " values out of "
                      << count << " }" << std::endl;
        }
        else
        {
            std::cerr << std::endl << "*               }" << std::endl;
        }
        return false;
    }
    return true;
}

#define INSTANTIATE(T)                                                                       \
    template bool rocgraph_clients_expect_array_near_tolerance(size_t      size,             \
                                                               const char* a_name,           \
                                                               const T* __restrict__ a,      \
                                                               size_t      a_inc,            \
                                                               const char* b_name,           \
                                                               const T* __restrict__ b,      \
                                                               size_t             b_inc,     \
                                                               floating_data_t<T> tol,       \
                                                               const char*        file_name, \
                                                               int32_t            file_line)
INSTANTIATE(float);
INSTANTIATE(double);

#undef INSTANTIATE
