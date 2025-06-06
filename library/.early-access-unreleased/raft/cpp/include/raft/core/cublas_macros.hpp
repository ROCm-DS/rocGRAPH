// Copyright (c) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/*
 * Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef __RAFT_RT_CUBLAS_MACROS_H
#define __RAFT_RT_CUBLAS_MACROS_H

#pragma once

#include <raft/core/error.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cublas_v2.h>
#else
#include <cublas_v2.h>
#endif

///@todo: enable this once we have logger enabled
// #include <cuml/common/logger.hpp>

#include <cstdint>

#define _CUBLAS_ERR_TO_STR(err) \
    case err:                   \
        return #err

namespace raft
{

    /**
 * @addtogroup error_handling
 * @{
 */

    /**
 * @brief Exception thrown when a cuBLAS error is encountered.
 */
    struct cublas_error : public raft::exception
    {
        explicit cublas_error(char const* const message)
            : raft::exception(message)
        {
        }
        explicit cublas_error(std::string const& message)
            : raft::exception(message)
        {
        }
    };

    /**
 * @}
 */

    namespace linalg
    {
        namespace detail
        {

            inline const char* cublas_error_to_string(cublasStatus_t err)
            {
                switch(err)
                {
                    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_SUCCESS);
                    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_NOT_INITIALIZED);
                    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_ALLOC_FAILED);
                    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_INVALID_VALUE);
                    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_ARCH_MISMATCH);
                    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_MAPPING_ERROR);
                    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_EXECUTION_FAILED);
                    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_INTERNAL_ERROR);
                    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_NOT_SUPPORTED);
                // FIXME(HIP/AMD): re-add when HIPBLAS_STATUS_LICENSE_ERROR is available
                //  (not available as of ROCm 6.1.2)
                //_CUBLAS_ERR_TO_STR(CUBLAS_STATUS_LICENSE_ERROR);
                default:
                    return "CUBLAS_STATUS_UNKNOWN";
                };
            }

        } // namespace detail
    } // namespace linalg
} // namespace raft

#undef _CUBLAS_ERR_TO_STR

/**
 * @addtogroup assertion
 * @{
 */

/**
 * @brief Error checking macro for cuBLAS runtime API functions.
 *
 * Invokes a cuBLAS runtime API function call, if the call does not return
 * CUBLAS_STATUS_SUCCESS, throws an exception detailing the cuBLAS error that occurred
 */
#define RAFT_CUBLAS_TRY(call)                                                    \
    do                                                                           \
    {                                                                            \
        cublasStatus_t const status = (call);                                    \
        if(CUBLAS_STATUS_SUCCESS != status)                                      \
        {                                                                        \
            std::string msg{};                                                   \
            SET_ERROR_MSG(msg,                                                   \
                          "cuBLAS error encountered at: ",                       \
                          "call='%s', Reason=%d:%s",                             \
                          #call,                                                 \
                          status,                                                \
                          raft::linalg::detail::cublas_error_to_string(status)); \
            throw raft::cublas_error(msg);                                       \
        }                                                                        \
    } while(0)

// FIXME: Remove after consumers rename
#ifndef CUBLAS_TRY
#define CUBLAS_TRY(call) RAFT_CUBLAS_TRY(call)
#endif

// /**
//  * @brief check for cuda runtime API errors but log error instead of raising
//  *        exception.
//  */
#define RAFT_CUBLAS_TRY_NO_THROW(call)                                     \
    do                                                                     \
    {                                                                      \
        cublasStatus_t const status = call;                                \
        if(CUBLAS_STATUS_SUCCESS != status)                                \
        {                                                                  \
            printf("CUBLAS call='%s' at file=%s line=%d failed with %s\n", \
                   #call,                                                  \
                   __FILE__,                                               \
                   __LINE__,                                               \
                   raft::linalg::detail::cublas_error_to_string(status));  \
        }                                                                  \
    } while(0)

/**
 * @}
 */
/** FIXME: remove after cuml rename */
#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call) CUBLAS_TRY(call)
#endif

/** FIXME: remove after cuml rename */
#ifndef CUBLAS_CHECK_NO_THROW
#define CUBLAS_CHECK_NO_THROW(call) RAFT_CUBLAS_TRY_NO_THROW(call)
#endif

#endif
