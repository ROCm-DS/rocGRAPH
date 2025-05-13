// Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <raft/amd_warp_primitives.h>

#include <rocprim/intrinsics/warp_shuffle.hpp>
#endif

// This file provides a few essential functions for use in __device__ code. The
// scope is necessarily limited to ensure that compilation times are minimized.
// Please make sure not to include large / expensive files from here.

namespace raft
{

/** helper macro for device inlined functions */
#define DI inline __device__
#define HDI inline __host__ __device__
#define HD __host__ __device__

    /**
 * @brief Provide a ceiling division operation ie. ceil(a / b)
 * @tparam IntType supposed to be only integers for now!
 */
    template <typename IntType>
    constexpr HDI IntType ceildiv(IntType a, IntType b)
    {
        return (a + b - 1) / b;
    }

    /**
 * @brief Provide an alignment function ie. ceil(a / b) * b
 * @tparam IntType supposed to be only integers for now!
 */
    template <typename IntType>
    constexpr HDI IntType alignTo(IntType a, IntType b)
    {
        return ceildiv(a, b) * b;
    }

    /**
 * @brief Provide an alignment function ie. (a / b) * b
 * @tparam IntType supposed to be only integers for now!
 */
    template <typename IntType>
    constexpr HDI IntType alignDown(IntType a, IntType b)
    {
        return (a / b) * b;
    }

    /**
 * @brief Check if the input is a power of 2
 * @tparam IntType data type (checked only for integers)
 */
    template <typename IntType>
    constexpr HDI bool isPo2(IntType num)
    {
        return (num && !(num & (num - 1)));
    }

    /**
 * @brief Give logarithm of the number to base-2
 * @tparam IntType data type (checked only for integers)
 */
    template <typename IntType>
    constexpr HDI IntType log2(IntType num, IntType ret = IntType(0))
    {
        return num <= IntType(1) ? ret : log2(num >> IntType(1), ++ret);
    }

    __device__ constexpr inline int WarpSize = []() {
#ifdef __HIP_PLATFORM_AMD__
        return hip_warp_primitives::WAVEFRONT_SIZE;
#else
        return 32;
#endif
    }();

    /** get the laneId of the current thread */
    DI int laneId()
    {
        int id;
#ifdef __HIP_PLATFORM_AMD__
        id = ::rocprim::lane_id();
#else
        asm("mov.s32 %0, %%laneid;" : "=r"(id));
#endif
        return id;
    }

    /** Device function to apply the input lambda across threads in the grid */
    template <int ItemsPerThread, typename L>
    DI void forEach(int num, L lambda)
    {
        int       idx        = (blockDim.x * blockIdx.x) + threadIdx.x;
        const int numThreads = blockDim.x * gridDim.x;
#pragma unroll
        for(int itr = 0; itr < ItemsPerThread; ++itr, idx += numThreads)
        {
            if(idx < num)
                lambda(idx, itr);
        }
    }

    /**
 * @brief Swap two values
 * @tparam T the datatype of the values
 * @param a first input
 * @param b second input
 */
    template <typename T>
    HDI void swapVals(T& a, T& b)
    {
        T tmp = a;
        a     = b;
        b     = tmp;
    }

} // namespace raft
