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
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/thirdparty/mdspan/include/experimental/mdspan>

#include <tuple>
#include <utility>

namespace raft::detail
{

    template <class T, std::size_t N, std::size_t... Idx>
    MDSPAN_INLINE_FUNCTION constexpr auto arr_to_tup(T (&arr)[N], std::index_sequence<Idx...>)
    {
        return std::make_tuple(arr[Idx]...);
    }

    template <class T, std::size_t N>
    MDSPAN_INLINE_FUNCTION constexpr auto arr_to_tup(T (&arr)[N])
    {
        return arr_to_tup(arr, std::make_index_sequence<N>{});
    }

    template <typename T>
    MDSPAN_INLINE_FUNCTION auto native_popc(T v) -> int32_t
    {
        int c = 0;
        for(; v != 0; v &= v - 1)
        {
            c++;
        }
        return c;
    }

    MDSPAN_INLINE_FUNCTION auto popc(uint32_t v) -> int32_t
    {
#if defined(__CUDA_ARCH__)
        return raft::__POPC(v);
#elif defined(__GNUC__) || defined(__clang__)
        return __builtin_popcount(v);
#else
        return native_popc(v);
#endif // compiler
    }

    MDSPAN_INLINE_FUNCTION auto popc(uint64_t v) -> int32_t
    {
#if defined(__CUDA_ARCH__)
        return __popcll(v);
#elif defined(__GNUC__) || defined(__clang__)
        return __builtin_popcountll(v);
#else
        return native_popc(v);
#endif // compiler
    }

} // end namespace raft::detail
