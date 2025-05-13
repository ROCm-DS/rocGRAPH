// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#ifdef __HIP_PLATFORM_AMD__
#include <raft/cuda_runtime.h>
#else
#include <cuda_runtime_api.h>
#endif

namespace raft
{
    /**
 * @brief Computes the population count (number of bits set to 1) in the provided integer.
 *
 * This is a templated device function that will be specialized for different integral types.
 * The return value type matches that of the underlying device builtin: `int`.
 *
 * @tparam T Integral type (e.g., int32_t, uint32_t, int64_t, uint64_t).
 * @param[in] v The integer for which to count set bits.
 * @return Number of bits set to 1 in \p v.
 */
    template <typename T>
    __device__ int __POPC(T v)
    {
        static_assert(!std::is_same_v<T, T>, "Invalid instantiation");
    }

    template <>
    __device__ inline int __POPC<int32_t>(int32_t v)
    {
        return __popc(v);
    }

    template <>
    __device__ inline int __POPC<int64_t>(int64_t v)
    {
        return __popcll(v);
    }

    template <>
    __device__ inline int __POPC<uint32_t>(uint32_t v)
    {
        return __popc(v);
    }

    template <>
    __device__ inline int __POPC<uint64_t>(uint64_t v)
    {
        return __popcll(v);
    }

    template <>
    __device__ inline int __POPC<unsigned long long int>(unsigned long long int v)
    {
        static_assert(sizeof(unsigned long long) == 8);
        return __popcll(v);
    }

    /**
 * \brief Find First Set
 * \return index of first set bit of lowest significance.
 * \note Return value type matches that of the underlying device builtin.
 * \note While `uint64_t` is defined as `unsigned long int` on x86_64,
 *        the HIP `__ffsll` device function provides `__ffsll` with `unsigned long long int`
 *        argument, which is also an 64-bit integer type on x86_64.
 *        However, the compilers typically see both as different types.
 *        We work with `uint64t` and `uint32t` here, so explicit instantiations
 *        for both are added here.
 */
    template <typename T>
    __device__ int __FFS(T v)
    {
        static_assert(!std::is_same_v<T, T>, "Invalid instantiation");
    }

    template <>
    __device__ inline int __FFS<int32_t>(int32_t v)
    {
        return __ffs(v);
    }

    template <>
    __device__ inline int __FFS<int64_t>(int64_t v)
    {
        return __ffsll(static_cast<unsigned long long int>(v));
    }

    template <>
    __device__ inline int __FFS<uint32_t>(uint32_t v)
    {
        return __ffs(v);
    }

    template <>
    __device__ inline int __FFS<uint64_t>(uint64_t v)
    {
        return __ffsll(static_cast<unsigned long long int>(v));
    }

} // namespace raft
