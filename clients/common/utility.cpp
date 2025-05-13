/*! \file */

// Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "utility.hpp"
#include "rocgraph_clients_envariables.hpp"

#include <chrono>
#include <cstdlib>
#include <fcntl.h>

/* ============================================================================================ */
// Return path of this executable
std::string rocgraph_exepath()
{
    std::string pathstr;
    char*       path = realpath("/proc/self/exe", 0);
    if(path)
    {
        char* p = strrchr(path, '/');
        if(p)
        {
            p[1]    = 0;
            pathstr = path;
        }
        free(path);
    }
    return pathstr;
}

/* ==================================================================================== */
// Return path where the test data file (rocgraph_test.data) is located
std::string rocgraph_datapath()
{
    // first check an environment variable
    if(rocgraph_clients_envariables::is_defined(rocgraph_clients_envariables::TEST_DATA_DIR))
    {
        return rocgraph_clients_envariables::get(rocgraph_clients_envariables::TEST_DATA_DIR);
    }

    std::string pathstr;
    std::string share_path = rocgraph_exepath() + "../share/rocgraph/test";
    char*       path       = realpath(share_path.c_str(), 0);
    if(path != NULL)
    {
        pathstr = path;
        pathstr += "/";
        free(path);
        return pathstr;
    }

    return rocgraph_exepath();
}

/* ============================================================================================ */
/*  timing:*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us(void)
{
    hipDeviceSynchronize();
    auto now = std::chrono::steady_clock::now();
    // struct timeval tv;
    // gettimeofday(&tv, NULL);
    //  return (tv.tv_sec * 1000 * 1000) + tv.tv_usec;
    auto duration
        = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return (static_cast<double>(duration));
};

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
double get_time_us_sync(hipStream_t stream)
{
    hipStreamSynchronize(stream);
    auto now = std::chrono::steady_clock::now();

    // struct timeval tv;
    // gettimeofday(&tv, NULL);
    // return (tv.tv_sec * 1000 * 1000) + tv.tv_usec;
    auto duration
        = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return (static_cast<double>(duration));
};
