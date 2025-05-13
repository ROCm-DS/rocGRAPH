/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#define CHECK_HIP_ERROR(ERROR)                    \
    do                                            \
    {                                             \
        auto error = ERROR;                       \
        if(error != hipSuccess)                   \
        {                                         \
            fprintf(stderr,                       \
                    "error: '%s'(%d) at %s:%d\n", \
                    hipGetErrorString(error),     \
                    error,                        \
                    __FILE__,                     \
                    __LINE__);                    \
            exit(EXIT_FAILURE);                   \
        }                                         \
    } while(0)

inline bool rocgraph_clients_expect_status(rocgraph_status status, rocgraph_status expect)
{
    if(status != expect)
    {
        std::cerr << "rocGRAPH status error: Expected " << rocgraph_status_to_string(expect)
                  << ", received " << rocgraph_status_to_string(status) << std::endl;
        if(expect == rocgraph_status_success)
        {
            exit(EXIT_FAILURE);
        }
    }
}

inline bool rocgraph_clients_expect_status_error(rocgraph_status   status,
                                                 rocgraph_status   expect,
                                                 rocgraph_error_t* error)
{
    if(status != expect)
    {
        std::cerr << "rocGRAPH status error: Expected " << rocgraph_status_to_string(expect)
                  << ", but received " << rocgraph_status_to_string(status) << ", '"
                  << rocgraph_error_message(error) << "'" << std::endl;
        exit(EXIT_FAILURE);
    }
}

#include "rocgraph_clients_are_eq.hpp"
#include "rocgraph_clients_expect_eq.hpp"
#include "rocgraph_clients_fail.hpp"

#include <iostream>

template <typename T>
bool rocgraph_clients_expect_eq(const char* a_name,
                                const T     a,
                                const char* b_name,
                                const T     b,
                                const char* file_name,
                                int32_t     file_line)
{
    const bool are_eq = rocgraph_clients_are_eq(a, b);
    if(are_eq == false)
    {
        std::cerr << "*** ERROR ROCGRAPH_CLIENTS_EXPECT_EQ(a,b)" << std::endl;
        std::cerr << "*   file: { " << std::endl;
        std::cerr << "*     name:  '" << file_name << "'," << std::endl;
        std::cerr << "*     line:  '" << file_line << "'" << std::endl;
        std::cerr << "*   }," << std::endl;
        std::cerr << "*   a:    { " << std::endl;
        std::cerr << "*     name:  '" << a_name << "'," << std::endl;
        std::cerr << "*     value: '" << a << "'" << std::endl;
        std::cerr << "*   }," << std::endl;
        std::cerr << "*   b:    { " << std::endl;
        std::cerr << "*     name:  '" << b_name << "'," << std::endl;
        std::cerr << "*     value: '" << b << "'" << std::endl;
        std::cerr << "*   }" << std::endl;
        return false;
    }
    return true;
}

#define INSTANTIATE(T)                                              \
    template bool rocgraph_clients_expect_eq(const char* a_name,    \
                                             const T     a,         \
                                             const char* b_name,    \
                                             const T     b,         \
                                             const char* file_name, \
                                             int32_t     file_line)

INSTANTIATE(bool);
INSTANTIATE(size_t);
INSTANTIATE(int32_t);
INSTANTIATE(int64_t);
INSTANTIATE(float);
INSTANTIATE(double);

#undef INSTANTIATE
