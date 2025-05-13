/*! \file */

/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "argdescr.h"
#include "message.h"
#include <hip/hip_runtime.h>
#include <iostream>

namespace rocgraph
{
    // Convert the current C++ exception to rocgraph_status
    // This allows extern "C" functions to return this function in a catch(...) block
    // while converting all C++ exceptions to an equivalent rocgraph_status here
    inline rocgraph_status exception_to_rocgraph_status(std::exception_ptr e
                                                        = std::current_exception())
    try
    {
        if(e)
            std::rethrow_exception(e);
        return rocgraph_status_success;
    }
    catch(const rocgraph_status& status)
    {
        return status;
    }
    catch(const std::bad_alloc&)
    {
        return rocgraph_status_memory_error;
    }
    catch(...)
    {
        return rocgraph_status_thrown_exception;
    }

}

/*******************************************************************************
 * Definitions
 * this file to not include any others
 * thereby it can include top-level definitions included by all
 ******************************************************************************/

//
// @brief Macros for coverage exclusion
//
#define ROCGRAPH_COV_EXCL_START (void)("LCOV_EXCL_START")
#define ROCGRAPH_COV_EXCL_STOP (void)("LCOV_EXCL_STOP")

namespace rocgraph
{
    /*******************************************************************************
 * \brief convert hipError_t to rocgraph_status
 ******************************************************************************/
    rocgraph_status get_rocgraph_status_for_hip_status(hipError_t status);
}

//
//
//
#define RETURN_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                                    \
    do                                                                                 \
    {                                                                                  \
        const hipError_t TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK);              \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                                         \
        {                                                                              \
            std::stringstream s;                                                       \
            s << "hip error detected: code '" << TMP_STATUS_FOR_CHECK << "', name '"   \
              << hipGetErrorName(TMP_STATUS_FOR_CHECK) << "', description '"           \
              << hipGetErrorString(TMP_STATUS_FOR_CHECK) << "'";                       \
            ROCGRAPH_ERROR_MESSAGE(                                                    \
                rocgraph::get_rocgraph_status_for_hip_status(TMP_STATUS_FOR_CHECK),    \
                s.str().c_str());                                                      \
            return rocgraph::get_rocgraph_status_for_hip_status(TMP_STATUS_FOR_CHECK); \
        }                                                                              \
    } while(false)

#define RETURN_WITH_MESSAGE_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK, MESSAGE)                   \
    do                                                                                      \
    {                                                                                       \
        const hipError_t TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK);                   \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                                              \
        {                                                                                   \
            std::stringstream s;                                                            \
            s << (MESSAGE) << ", hip error detected: code '" << TMP_STATUS_FOR_CHECK        \
              << "', name '" << hipGetErrorName(TMP_STATUS_FOR_CHECK) << "', description '" \
              << hipGetErrorString(TMP_STATUS_FOR_CHECK) << "'";                            \
            ROCGRAPH_ERROR_MESSAGE(                                                         \
                rocgraph::get_rocgraph_status_for_hip_status(TMP_STATUS_FOR_CHECK),         \
                s.str().c_str());                                                           \
            return rocgraph::get_rocgraph_status_for_hip_status(TMP_STATUS_FOR_CHECK);      \
        }                                                                                   \
    } while(false)

//
//
//

#define THROW_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                                              \
    do                                                                                          \
    {                                                                                           \
        const hipError_t TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK);                       \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                                                  \
        {                                                                                       \
            std::stringstream s;                                                                \
            s << "throwing exception due to hip error detected: code '" << TMP_STATUS_FOR_CHECK \
              << "', name '" << hipGetErrorName(TMP_STATUS_FOR_CHECK) << "', description '"     \
              << hipGetErrorString(TMP_STATUS_FOR_CHECK) << "'";                                \
            ROCGRAPH_ERROR_MESSAGE(                                                             \
                rocgraph::get_rocgraph_status_for_hip_status(TMP_STATUS_FOR_CHECK),             \
                s.str().c_str());                                                               \
            throw rocgraph::get_rocgraph_status_for_hip_status(TMP_STATUS_FOR_CHECK);           \
        }                                                                                       \
    } while(false)

#define THROW_WITH_MESSAGE_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK, MESSAGE)                      \
    do                                                                                        \
    {                                                                                         \
        const hipError_t TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK);                     \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                                                \
        {                                                                                     \
            std::stringstream s;                                                              \
            s << (MESSAGE) << ", throwing exception due to hip error detected: code '"        \
              << TMP_STATUS_FOR_CHECK << "', name '" << hipGetErrorName(TMP_STATUS_FOR_CHECK) \
              << "', description '" << hipGetErrorString(TMP_STATUS_FOR_CHECK) << "'";        \
            ROCGRAPH_ERROR_MESSAGE(                                                           \
                rocgraph::get_rocgraph_status_for_hip_status(TMP_STATUS_FOR_CHECK),           \
                s.str().c_str());                                                             \
            throw rocgraph::get_rocgraph_status_for_hip_status(TMP_STATUS_FOR_CHECK);         \
        }                                                                                     \
    } while(false)

//
//
//
#define WARNING_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                      \
    do                                                                    \
    {                                                                     \
        const hipError_t TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK); \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                            \
        {                                                                 \
            ROCGRAPH_WARNING_MESSAGE("hip error detected");               \
        }                                                                 \
    } while(false)

#define FORWARD_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                                 \
    do                                                                               \
    {                                                                                \
        const hipError_t TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK);            \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                                       \
        {                                                                            \
            std::stringstream s;                                                     \
            s << "hip error detected: code '" << TMP_STATUS_FOR_CHECK << "', name '" \
              << hipGetErrorName(TMP_STATUS_FOR_CHECK) << "', description '"         \
              << hipGetErrorString(TMP_STATUS_FOR_CHECK) << "'";                     \
            ROCGRAPH_ERROR_MESSAGE(                                                  \
                rocgraph::get_rocgraph_status_for_hip_status(TMP_STATUS_FOR_CHECK),  \
                s.str().c_str());                                                    \
            return TMP_STATUS_FOR_CHECK;                                             \
        }                                                                            \
    } while(false)

#define FORWARD_WITH_MESSAGE_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK, MESSAGE)                  \
    do                                                                                      \
    {                                                                                       \
        const hipError_t TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK);                   \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                                              \
        {                                                                                   \
            std::stringstream s;                                                            \
            s << (MESSAGE) << ", hip error detected: code '" << TMP_STATUS_FOR_CHECK        \
              << "', name '" << hipGetErrorName(TMP_STATUS_FOR_CHECK) << "', description '" \
              << hipGetErrorString(TMP_STATUS_FOR_CHECK) << "'";                            \
            ROCGRAPH_ERROR_MESSAGE(                                                         \
                rocgraph::get_rocgraph_status_for_hip_status(TMP_STATUS_FOR_CHECK),         \
                s.str().c_str());                                                           \
            return TMP_STATUS_FOR_CHECK;                                                    \
        }                                                                                   \
    } while(false)

//
//
//

#define RETURN_IF_ROCGRAPH_ERROR(INPUT_STATUS_FOR_CHECK)                       \
    do                                                                         \
    {                                                                          \
        const rocgraph_status TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK); \
        if(TMP_STATUS_FOR_CHECK != rocgraph_status_success)                    \
        {                                                                      \
            ROCGRAPH_ERROR_MESSAGE(TMP_STATUS_FOR_CHECK, "none");              \
            return TMP_STATUS_FOR_CHECK;                                       \
        }                                                                      \
    } while(false)

//
//
//
#define THROW_IF_ROCGRAPH_ERROR(INPUT_STATUS_FOR_CHECK)                            \
    do                                                                             \
    {                                                                              \
        const rocgraph_status TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK);     \
        if(TMP_STATUS_FOR_CHECK != rocgraph_status_success)                        \
        {                                                                          \
            ROCGRAPH_ERROR_MESSAGE(TMP_STATUS_FOR_CHECK,                           \
                                   "rocGRAPH error detected, throwing exception"); \
            throw TMP_STATUS_FOR_CHECK;                                            \
        }                                                                          \
    } while(false)

//
//
//
#define RETURN_WITH_MESSAGE_IF_ROCGRAPH_ERROR(INPUT_STATUS_FOR_CHECK, MSG)     \
    do                                                                         \
    {                                                                          \
        const rocgraph_status TMP_STATUS_FOR_CHECK = (INPUT_STATUS_FOR_CHECK); \
        if(TMP_STATUS_FOR_CHECK != rocgraph_status_success)                    \
        {                                                                      \
            ROCGRAPH_ERROR_MESSAGE(TMP_STATUS_FOR_CHECK, MSG);                 \
            return TMP_STATUS_FOR_CHECK;                                       \
        }                                                                      \
    } while(false)

//
//
//
#define RETURN_ROCGRAPH_EXCEPTION()                                                  \
    do                                                                               \
    {                                                                                \
        const rocgraph_status TMP_STATUS = rocgraph::exception_to_rocgraph_status(); \
        ROCGRAPH_ERROR_MESSAGE(TMP_STATUS, "exception detected");                    \
        return TMP_STATUS;                                                           \
    } while(false)

#define THROW_IF_HIPLAUNCHKERNELGGL_ERROR(...)                                                 \
    do                                                                                         \
    {                                                                                          \
        if(false == rocgraph_debug_variables.get_debug_kernel_launch())                        \
        {                                                                                      \
            hipLaunchKernelGGL(__VA_ARGS__);                                                   \
        }                                                                                      \
        else                                                                                   \
        {                                                                                      \
            THROW_WITH_MESSAGE_IF_HIP_ERROR(hipGetLastError(), "prior to hipLaunchKernelGGL"); \
            hipLaunchKernelGGL(__VA_ARGS__);                                                   \
            THROW_IF_HIP_ERROR(hipGetLastError());                                             \
        }                                                                                      \
    } while(false)

#define RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(...)                                                 \
    do                                                                                          \
    {                                                                                           \
        if(false == rocgraph_debug_variables.get_debug_kernel_launch())                         \
        {                                                                                       \
            hipLaunchKernelGGL(__VA_ARGS__);                                                    \
        }                                                                                       \
        else                                                                                    \
        {                                                                                       \
            RETURN_WITH_MESSAGE_IF_HIP_ERROR(hipGetLastError(), "prior to hipLaunchKernelGGL"); \
            hipLaunchKernelGGL(__VA_ARGS__);                                                    \
            RETURN_IF_HIP_ERROR(hipGetLastError());                                             \
        }                                                                                       \
    } while(false)

#define RETURN_ROCGRAPH_ERROR_IF(STATUS, INPUT_STATUS_FOR_CHECK)     \
    do                                                               \
    {                                                                \
        if(INPUT_STATUS_FOR_CHECK)                                   \
        {                                                            \
            ROCGRAPH_ERROR_MESSAGE(STATUS, #INPUT_STATUS_FOR_CHECK); \
            return STATUS;                                           \
        }                                                            \
    } while(false)

#define PRINT_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                                   \
    {                                                                                \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;                    \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                                       \
        {                                                                            \
            std::stringstream s;                                                     \
            s << "hip error detected: code '" << TMP_STATUS_FOR_CHECK << "', name '" \
              << hipGetErrorName(TMP_STATUS_FOR_CHECK) << "', description '"         \
              << hipGetErrorString(TMP_STATUS_FOR_CHECK) << "'";                     \
            ROCGRAPH_ERROR_MESSAGE(                                                  \
                rocgraph::get_rocgraph_status_for_hip_status(TMP_STATUS_FOR_CHECK),  \
                s.str().c_str());                                                    \
        }                                                                            \
    }
