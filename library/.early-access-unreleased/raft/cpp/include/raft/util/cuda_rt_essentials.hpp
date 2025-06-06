// Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

// This file provides a few essential functions that wrap the CUDA runtime API.
// The scope is necessarily limited to ensure that compilation times are
// minimized. Please make sure not to include large / expensive files from here.

#include <raft/core/error.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include <cstdio>

namespace raft
{

    /**
 * @brief Exception thrown when a CUDA error is encountered.
 */
    struct cuda_error : public raft::exception
    {
        explicit cuda_error(char const* const message)
            : raft::exception(message)
        {
        }
        explicit cuda_error(std::string const& message)
            : raft::exception(message)
        {
        }
    };

} // namespace raft

/**
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call, if the call does not return
 * cudaSuccess, invokes cudaGetLastError() to clear the error and throws an
 * exception detailing the CUDA error that occurred
 *
 */
#define RAFT_CUDA_TRY(call)                              \
    do                                                   \
    {                                                    \
        cudaError_t const status = call;                 \
        if(status != cudaSuccess)                        \
        {                                                \
            static_cast<void>(cudaGetLastError());       \
            std::string msg{};                           \
            SET_ERROR_MSG(msg,                           \
                          "CUDA error encountered at: ", \
                          "call='%s', Reason=%s:%s",     \
                          #call,                         \
                          cudaGetErrorName(status),      \
                          cudaGetErrorString(status));   \
            throw raft::cuda_error(msg);                 \
        }                                                \
    } while(0)

/**
 * @brief Debug macro to check for CUDA errors
 *
 * In a non-release build, this macro will synchronize the specified stream
 * before error checking. In both release and non-release builds, this macro
 * checks for any pending CUDA errors from previous calls. If an error is
 * reported, an exception is thrown detailing the CUDA error that occurred.
 *
 * The intent of this macro is to provide a mechanism for synchronous and
 * deterministic execution for debugging asynchronous CUDA execution. It should
 * be used after any asynchronous CUDA call, e.g., cudaMemcpyAsync, or an
 * asynchronous kernel launch.
 */
#ifndef NDEBUG
#define RAFT_CHECK_CUDA(stream) RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
#else
#define RAFT_CHECK_CUDA(stream) RAFT_CUDA_TRY(cudaPeekAtLastError());
#endif

// /**
//  * @brief check for cuda runtime API errors but log error instead of raising
//  *        exception.
//  */
#define RAFT_CUDA_TRY_NO_THROW(call)                                     \
    do                                                                   \
    {                                                                    \
        cudaError_t const status = call;                                 \
        if(cudaSuccess != status)                                        \
        {                                                                \
            printf("CUDA call='%s' at file=%s line=%d failed with %s\n", \
                   #call,                                                \
                   __FILE__,                                             \
                   __LINE__,                                             \
                   cudaGetErrorString(status));                          \
        }                                                                \
    } while(0)
