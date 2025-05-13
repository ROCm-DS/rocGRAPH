/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#include "auto_testing_bad_arg.hpp"
#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "rocgraph_check.hpp"

#include "rocgraph_clients_create_test_graph.hpp"
#include "rocgraph_matrix_factory.hpp"
#include "rocgraph_vector_utils.hpp"
#include "utility.hpp"
#include <rocgraph.hpp>
template <typename T>
inline T* rocgraph_fake_pointer()
{
    return static_cast<T*>((void*)0x4);
}

template <typename T>
inline T rocgraph_nan()
{
    return std::numeric_limits<T>::quiet_NaN();
}

template <typename T>
inline T rocgraph_inf()
{
    return std::numeric_limits<T>::infinity();
}

template <typename T>
floating_data_t<T> get_near_check_tol(const Arguments& arg)
{
    return static_cast<floating_data_t<T>>(arg.tolm) * rocgraph_clients_default_tolerance<T>::value;
}

//
// Compute gflops
//

inline double get_gpu_gflops(double gpu_time_used, double gflop_count)
{
    return gflop_count / gpu_time_used * 1e6;
}

template <typename F, typename... Ts>
inline double get_gpu_gflops(double gpu_time_used, F count, Ts... ts)
{
    return get_gpu_gflops(gpu_time_used, count(ts...));
}

//
// Compute gbyte
//
inline double get_gpu_gbyte(double gpu_time_used, double gbyte_count)
{
    return gbyte_count / gpu_time_used * 1e6;
}

template <typename F, typename... Ts>
inline double get_gpu_gbyte(double gpu_time_used, F count, Ts... ts)
{
    return get_gpu_gbyte(gpu_time_used, count(ts...));
}

inline double get_gpu_time_msec(double gpu_time_used)
{
    return gpu_time_used / 1e3;
}

/*  Check hmm availability
    Not used anywhere and causes a LOT of warnings because return value of
    hipGetDevice and hipDeviceGetAttribute is not checked.
*/

#if 0
inline bool is_hmm_enabled()
{
    int deviceID, hmm_enabled;
    hipGetDevice(&deviceID);
    hipDeviceGetAttribute(&hmm_enabled, hipDeviceAttributeManagedMemory, deviceID);

    return hmm_enabled;
}
#endif

template <typename First>
void testing_dispatch_extra_names(const Arguments& arg,
                                  const char*      name,
                                  const char*      fname,
                                  First            f)
{
    std::cout << " " << fname << std::endl;
}

template <typename First, typename... P>
void testing_dispatch_extra_names(
    const Arguments& arg, const char* name, const char* fname, First f, P... p)
{
    std::cout << " " << fname << std::endl;
    return testing_dispatch_extra_names(arg, name, p...);
}

template <typename First>
void testing_dispatch_extra_all(const Arguments& arg, const char* fname, First f)
{
    f(arg);
}

template <typename First, typename... P>
void testing_dispatch_extra_all(const Arguments& arg, const char* fname, First f, P... p)
{
    f(arg);
    return testing_dispatch_extra_all(arg, p...);
}

template <typename First>
bool testing_dispatch_extra_select(const Arguments& arg,
                                   const char*      name,
                                   const char*      fname,
                                   First            f)
{
    if(!strcmp(fname, name))
    {
        f(arg);
        return true;
    }
    else
    {
        return false;
    }
}

template <typename First, typename... P>
bool testing_dispatch_extra_select(
    const Arguments& arg, const char* name, const char* fname, First f, P... p)
{
    if(!strcmp(fname, arg.name))
    {
        f(arg);
        return true;
    }
    else
    {
        return testing_dispatch_extra_select(arg, name, p...);
    }
}

template <typename... P>
void testing_dispatch_extra(const Arguments& arg, P... p)
{
    if(!strcmp(arg.name, arg.function))
    {
        testing_dispatch_extra_all(arg, p...);
    }
    else
    {
        if(!testing_dispatch_extra_select(arg, arg.name, p...))
        {
            std::cerr << "name is invalid '" << arg.name << "'" << std::endl;
            std::cerr << "list of test names is:" << std::endl;
            testing_dispatch_extra_names(arg, arg.name, p...);
            ROCGRAPH_CLIENTS_FAIL();
        }
    }
}
