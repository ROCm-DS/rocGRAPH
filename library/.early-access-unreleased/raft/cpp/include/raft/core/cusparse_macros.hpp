// Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#pragma once

#include <raft/core/error.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cusparse.h>
#else
#include <cusparse.h>
#endif

///@todo: enable this once logging is enabled
// #include <cuml/common/logger.hpp>

#define _CUSPARSE_ERR_TO_STR(err) \
    case err:                     \
        return #err;

// Notes:
//(1.) CUDA_VER_10_1_UP aggregates all the CUDA version selection logic;
//(2.) to enforce a lower version,
//
//`#define CUDA_ENFORCE_LOWER
// #include <raft/sparse/detail/cusparse_wrappers.h>`
//
// (i.e., before including this header)
//
#define CUDA_VER_10_1_UP (CUDART_VERSION >= 10100)
#if defined(__HIP_PLATFORM_AMD__)
#undef CUDA_VER_10_1_UP
// The selection of what subset of cusparse API's are made available is based on pre-processor logic
// around whether "CUDA_VER_10_1_UP" is defined or not. When compiling with the HIP/AMD toolchain we
// force the latest version of the cuparse API's as these are known to be supported.
#define CUDA_VER_10_1_UP 1
#endif

namespace raft
{

    /**
 * @ingroup error_handling
 * @{
 */

    /**
 * @brief Exception thrown when a cuSparse error is encountered.
 */
    struct cusparse_error : public raft::exception
    {
        explicit cusparse_error(char const* const message)
            : raft::exception(message)
        {
        }
        explicit cusparse_error(std::string const& message)
            : raft::exception(message)
        {
        }
    };

    /**
 * @}
 */
    namespace sparse
    {
        namespace detail
        {

            inline const char* cusparse_error_to_string(cusparseStatus_t err)
            {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 10100
                return cusparseGetErrorString(err);
#else // CUDART_VERSION
                switch(err)
                {
                    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_SUCCESS);
                    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_NOT_INITIALIZED);
                    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_ALLOC_FAILED);
                    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_INVALID_VALUE);
                    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_ARCH_MISMATCH);
                    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_EXECUTION_FAILED);
                    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_INTERNAL_ERROR);
                    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
                default:
                    return "CUSPARSE_STATUS_UNKNOWN";
                };
#endif // CUDART_VERSION
            }

        } // namespace detail
    } // namespace sparse
} // namespace raft

#undef _CUSPARSE_ERR_TO_STR

/**
 * @ingroup assertion
 * @{
 */

/**
 * @brief Error checking macro for cuSparse runtime API functions.
 *
 * Invokes a cuSparse runtime API function call, if the call does not return
 * CUSPARSE_STATUS_SUCCESS, throws an exception detailing the cuSparse error that occurred
 */
#define RAFT_CUSPARSE_TRY(call)                                                    \
    do                                                                             \
    {                                                                              \
        cusparseStatus_t const status = (call);                                    \
        if(CUSPARSE_STATUS_SUCCESS != status)                                      \
        {                                                                          \
            std::string msg{};                                                     \
            SET_ERROR_MSG(msg,                                                     \
                          "cuSparse error encountered at: ",                       \
                          "call='%s', Reason=%d:%s",                               \
                          #call,                                                   \
                          status,                                                  \
                          raft::sparse::detail::cusparse_error_to_string(status)); \
            throw raft::cusparse_error(msg);                                       \
        }                                                                          \
    } while(0)

/**
 * @}
 */

// FIXME: Remove after consumer rename
#ifndef CUSPARSE_TRY
#define CUSPARSE_TRY(call) RAFT_CUSPARSE_TRY(call)
#endif

// FIXME: Remove after consumer rename
#ifndef CUSPARSE_CHECK
#define CUSPARSE_CHECK(call) CUSPARSE_TRY(call)
#endif

/**
 * @ingroup assertion
 * @{
 */
//@todo: use logger here once logging is enabled
/** check for cusparse runtime API errors but do not assert */
#define RAFT_CUSPARSE_TRY_NO_THROW(call)                                 \
    do                                                                   \
    {                                                                    \
        cusparseStatus_t err = call;                                     \
        if(err != CUSPARSE_STATUS_SUCCESS)                               \
        {                                                                \
            printf("CUSPARSE call='%s' got errorcode=%d err=%s",         \
                   #call,                                                \
                   err,                                                  \
                   raft::sparse::detail::cusparse_error_to_string(err)); \
        }                                                                \
    } while(0)

/**
 * @}
 */

// FIXME: Remove after consumer rename
#ifndef CUSPARSE_CHECK_NO_THROW
#define CUSPARSE_CHECK_NO_THROW(call) RAFT_CUSPARSE_TRY_NO_THROW(call)
#endif
