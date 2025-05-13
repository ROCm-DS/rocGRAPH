// Copyright (c) 2022, NVIDIA CORPORATION.
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

#ifdef __HIP_PLATFORM_AMD__
#include <raft/curand.h>
#else
#include <curand.h>
#endif

namespace raft::random
{
    namespace detail
    {

// @todo: We probably want to scrape through and replace any consumers of
// these wrappers with our RNG
/** check for curand runtime API errors and assert accordingly */
#define CURAND_CHECK(call)                            \
    do                                                \
    {                                                 \
        curandStatus_t status = call;                 \
        ASSERT(status == CURAND_STATUS_SUCCESS,       \
               "FAIL: curand-call='%s'. Reason:%d\n", \
               #call,                                 \
               status);                               \
    } while(0)

        /**
 * @defgroup normal curand normal random number generation operations
 * @{
 */
        template <typename T>
        curandStatus_t curandGenerateNormal(
            curandGenerator_t generator, T* outputPtr, size_t n, T mean, T stddev);

        template <>
        inline curandStatus_t curandGenerateNormal(
            curandGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev)
        {
            return curandGenerateNormal(generator, outputPtr, n, mean, stddev);
        }

        template <>
        inline curandStatus_t curandGenerateNormal(
            curandGenerator_t generator, double* outputPtr, size_t n, double mean, double stddev)
        {
            return curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev);
        }
        /** @} */

    }; // end namespace detail
}; // end namespace raft::random
