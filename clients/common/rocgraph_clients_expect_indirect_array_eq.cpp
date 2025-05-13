/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_clients_expect_indirect_array_eq.hpp"
#include "rocgraph_clients_are_eq.hpp"
#include "rocgraph_clients_fail.hpp"
#include <iostream>

template <typename T, typename I>
bool rocgraph_clients_expect_indirect_array_eq(size_t      size,
                                               const char* a_name,
                                               const T* __restrict__ a,
                                               size_t      a_inc,
                                               const char* a_perm_name,
                                               const I* __restrict__ a_perm,
                                               const char* b_name,
                                               const T* __restrict__ b,
                                               size_t      b_inc,
                                               const char* b_perm_name,
                                               const I* __restrict__ b_perm,
                                               const char* file_name,
                                               int32_t     file_line)
{
    static constexpr size_t s_max_count = 10;
    size_t                  indices[s_max_count];
    size_t                  count = 0;
    for(size_t i = 0; i < size; ++i)
    {
        const I    a_i    = (a_perm != nullptr) ? a_perm[i] : i;
        const I    b_i    = (b_perm != nullptr) ? b_perm[i] : i;
        const bool are_eq = rocgraph_clients_are_eq(a[a_i * a_inc], b[b_i * b_inc]);
        if(are_eq == false)
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
        std::cerr << "*** ERROR ROCGRAPH_CLIENTS_EXPECT_INDIRECT_ARRAY_EQ(size, a, a_inc, a_perm, "
                     "b, b_inc, b_perm)"
                  << std::endl;
        std::cerr << "*   file: { " << std::endl;
        std::cerr << "*     name:  '" << file_name << "'," << std::endl;
        std::cerr << "*     line:  '" << file_line << "'" << std::endl;
        std::cerr << "*   }," << std::endl;
        std::cerr << "*   a:    { " << std::endl;
        std::cerr << "*     name:  '" << a_name << "'," << std::endl;
        std::cerr << "*     inc:  '" << a_inc << "'," << std::endl;
        std::cerr << "*     perm:  '" << a_perm_name << "'," << std::endl;
        std::cerr << "*   }," << std::endl;
        std::cerr << "*   b:    { " << std::endl;
        std::cerr << "*     name:  '" << b_name << "'," << std::endl;
        std::cerr << "*     inc:   '" << b_inc << "'," << std::endl;
        std::cerr << "*     perm:  '" << b_perm_name << "'," << std::endl;
        std::cerr << "*   }," << std::endl;
        std::cerr << "*   values: { " << std::endl;
        for(size_t i = 0; i < std::min(s_max_count, count); ++i)
        {
            const size_t index   = indices[i];
            const size_t a_index = (a_perm != nullptr) ? a_perm[index] : index;
            const size_t b_index = (b_perm != nullptr) ? b_perm[index] : index;
            const auto   a_val   = a[a_index * a_inc];
            const auto   b_val   = b[b_index * b_inc];
            if(i > 0)
            {
                std::cerr << "," << std::endl;
            }
            std::cerr << "*     { index: '" << index << "'," << std::endl;
            std::cerr << "*       a[" << a_index << "]: '" << a_val << "'," << std::endl;
            std::cerr << "*       b[" << b_index << "]: '" << b_val << "'}";
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

#define INSTANTIATE(T, I)                                                                 \
    template bool rocgraph_clients_expect_indirect_array_eq(size_t      size,             \
                                                            const char* a_name,           \
                                                            const T* __restrict__ a,      \
                                                            size_t      a_inc,            \
                                                            const char* a_perm_name,      \
                                                            const I* __restrict__ a_perm, \
                                                            const char* b_name,           \
                                                            const T* __restrict__ b,      \
                                                            size_t      b_inc,            \
                                                            const char* b_perm_name,      \
                                                            const I* __restrict__ b_perm, \
                                                            const char* file_name,        \
                                                            int32_t     file_line)
INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(float, int32_t);
INSTANTIATE(double, int32_t);
INSTANTIATE(size_t, int32_t);
#undef INSTANTIATE
