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

#ifndef __RAFT_RT_CUSOLVER_MACROS_H
#define __RAFT_RT_CUSOLVER_MACROS_H

#pragma once

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cusolver.h>
#else
#include <cusolverDn.h>
#include <cusolverSp.h>
#endif

///@todo: enable this once logging is enabled
// #include <cuml/common/logger.hpp>
#include <raft/util/cudart_utils.hpp>

#include <type_traits>

#define _CUSOLVER_ERR_TO_STR(err) \
    case err:                     \
        return #err;

namespace raft
{

    /**
 * @ingroup error_handling
 * @{
 */

    /**
 * @brief Exception thrown when a cuSOLVER error is encountered.
 */
    struct cusolver_error : public raft::exception
    {
        explicit cusolver_error(char const* const message)
            : raft::exception(message)
        {
        }
        explicit cusolver_error(std::string const& message)
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

            inline const char* cusolver_error_to_string(cusolverStatus_t err)
            {
                switch(err)
                {
                    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_SUCCESS);
                    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_NOT_INITIALIZED);
                    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_ALLOC_FAILED);
                    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_INVALID_VALUE);
                    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_ARCH_MISMATCH);
                    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_EXECUTION_FAILED);
                    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_INTERNAL_ERROR);
                    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
                    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_ZERO_PIVOT);
                    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_NOT_SUPPORTED);
                default:
                    return "CUSOLVER_STATUS_UNKNOWN";
                };
            }

        } // namespace detail
    } // namespace linalg
} // namespace raft

#undef _CUSOLVER_ERR_TO_STR

/**
 * @ingroup assertion
 * @{
 */

/**
 * @brief Error checking macro for cuSOLVER runtime API functions.
 *
 * Invokes a cuSOLVER runtime API function call, if the call does not return
 * CUSolver_STATUS_SUCCESS, throws an exception detailing the cuSOLVER error that occurred
 */
#define RAFT_CUSOLVER_TRY(call)                                                    \
    do                                                                             \
    {                                                                              \
        cusolverStatus_t const status = (call);                                    \
        if(CUSOLVER_STATUS_SUCCESS != status)                                      \
        {                                                                          \
            std::string msg{};                                                     \
            SET_ERROR_MSG(msg,                                                     \
                          "cuSOLVER error encountered at: ",                       \
                          "call='%s', Reason=%d:%s",                               \
                          #call,                                                   \
                          status,                                                  \
                          raft::linalg::detail::cusolver_error_to_string(status)); \
            throw raft::cusolver_error(msg);                                       \
        }                                                                          \
    } while(0)

// FIXME: remove after consumer rename
#ifndef CUSOLVER_TRY
#define CUSOLVER_TRY(call) RAFT_CUSOLVER_TRY(call)
#endif

// /**
//  * @brief check for cuda runtime API errors but log error instead of raising
//  *        exception.
//  */
#define RAFT_CUSOLVER_TRY_NO_THROW(call)                                     \
    do                                                                       \
    {                                                                        \
        cusolverStatus_t const status = call;                                \
        if(CUSOLVER_STATUS_SUCCESS != status)                                \
        {                                                                    \
            printf("CUSOLVER call='%s' at file=%s line=%d failed with %s\n", \
                   #call,                                                    \
                   __FILE__,                                                 \
                   __LINE__,                                                 \
                   raft::linalg::detail::cusolver_error_to_string(status));  \
        }                                                                    \
    } while(0)

/**
 * @}
 */

// FIXME: remove after cuml rename
#ifndef CUSOLVER_CHECK
#define CUSOLVER_CHECK(call) CUSOLVER_TRY(call)
#endif

#ifndef CUSOLVER_CHECK_NO_THROW
#define CUSOLVER_CHECK_NO_THROW(call) CUSOLVER_TRY_NO_THROW(call)
#endif

#endif
