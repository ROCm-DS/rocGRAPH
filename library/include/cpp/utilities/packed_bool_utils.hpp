// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*
 * Copyright (C) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "thrust_tuple_utils.hpp"
#include <raft/util/cudart_utils.hpp>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/tuple.h>

#include <limits>
#include <type_traits>
#include <utility>

#ifdef ROCGRAPH_USE_WARPSIZE_32
typedef uint32_t packed_bool_container_t;
#else
typedef uint64_t packed_bool_container_t;
#endif

namespace rocgraph
{
    namespace detail
    {

        template <typename ValueIterator, typename value_t, std::size_t... Is>
        constexpr std::enable_if_t<
            rocgraph::is_thrust_tuple_of_arithmetic<
                typename thrust::iterator_traits<ValueIterator>::value_type>::value
                && rocgraph::is_thrust_tuple_of_arithmetic<value_t>::value,
            bool>
            has_packed_bool_element(std::index_sequence<Is...>)
        {
            static_assert(thrust::tuple_size<
                              typename thrust::iterator_traits<ValueIterator>::value_type>::value
                          == thrust::tuple_size<value_t>::value);
            return (
                ...
                || (std::is_same_v<
                        typename thrust::tuple_element<
                            Is,
                            typename thrust::iterator_traits<ValueIterator>::value_type>::type,
                        packed_bool_container_t>
                    && std::is_same_v<typename thrust::tuple_element<Is, value_t>::type, bool>));
        }
    } // namespace detail

    template <typename T>
    __device__ std::enable_if_t<sizeof(T) == 8, int32_t> bit_count(T value)
    {
        return __popcll(value);
    }

    template <typename T>
    __device__ std::enable_if_t<sizeof(T) == 4, int32_t> bit_count(T value)
    {
        return __popc(value);
    }

    /*  There is no __fns / __fnsll in rocm, we have __fns64/__fns32
        https://rocm.docs.amd.com/projects/HIP/en/docs-6.3.3/reference/math_api.html
    */
    template <typename T>
    __device__ std::enable_if_t<sizeof(T) == 8, int32_t>
               n_set_bit(T mask, uint32_t base, int32_t offset)
    {
        return __fns64(mask, base, offset);
    }

    template <typename T>
    __device__ std::enable_if_t<sizeof(T) == 4, int32_t>
               n_set_bit(T mask, uint32_t base, int32_t offset)
    {
        return __fns32(mask, base, offset);
    }

    template <typename ValueIterator, typename value_t>
    constexpr bool is_packed_bool()
    {
        return std::is_same_v<typename thrust::iterator_traits<ValueIterator>::value_type,
                              packed_bool_container_t>
               && std::is_same_v<value_t, bool>;
    }

    template <typename ValueIterator, typename value_t>
    constexpr bool has_packed_bool_element()
    {
        static_assert(
            (std::is_arithmetic_v<typename thrust::iterator_traits<ValueIterator>::value_type>
             && std::is_arithmetic_v<value_t>)
            || (rocgraph::is_thrust_tuple_of_arithmetic<
                    typename thrust::iterator_traits<ValueIterator>::value_type>::value
                && rocgraph::is_thrust_tuple_of_arithmetic<value_t>::value));

        if constexpr(std::is_arithmetic_v<
                         typename thrust::iterator_traits<ValueIterator>::value_type>
                     && std::is_arithmetic_v<value_t>)
        {
            return std::is_same_v<typename thrust::iterator_traits<ValueIterator>::value_type,
                                  packed_bool_container_t>
                   && std::is_same_v<value_t, bool>;
        }
        else
        {
            static_assert(thrust::tuple_size<
                              typename thrust::iterator_traits<ValueIterator>::value_type>::value
                          == thrust::tuple_size<value_t>::value);
            return detail::has_packed_bool_element<ValueIterator, value_t>(
                std::make_index_sequence<thrust::tuple_size<value_t>::value>());
        }
    }

    /* This will return the number of bits in a packed_bool_container_t */
    constexpr size_t packed_bools_per_word()
    {
#ifdef ROCGRAPH_USE_WARPSIZE_32
        return 32u;
#else
        return 64u;
#endif
    }

    /*
    This will return the number of packed_bool_container_t elements necessery to store
    a bool_size number of bits */
    constexpr size_t packed_bool_size(size_t bool_size)
    {
        return (bool_size + (packed_bools_per_word() - 1)) / packed_bools_per_word();
    }

    /*
    This will return a mask for the packed_bool_container_t that contains the bit for bool_offset.
    Only one bit is set on this mask. */
    template <typename T>
    constexpr packed_bool_container_t packed_bool_mask(T bool_offset)
    {
        return packed_bool_container_t{1} << (bool_offset % packed_bools_per_word());
    }

    constexpr packed_bool_container_t packed_bool_full_mask()
    {
#ifdef ROCGRAPH_USE_WARPSIZE_32
        return UINT32_MAX;
#else
        return UINT64_MAX;
#endif
    }

    template <typename T>
    constexpr packed_bool_container_t packed_bool_partial_mask(T num_set_bits)
    {
        return packed_bool_full_mask() >> (packed_bools_per_word() - num_set_bits);
    }

    constexpr packed_bool_container_t packed_bool_empty_mask()
    {
        return packed_bool_container_t{0};
    }

    /*
    This will return the zero based index of the packed_bool_container_t in the array
    that container the bool_offset bit */
    template <typename T>
    constexpr T packed_bool_offset(T bool_offset)
    {
        return bool_offset / packed_bools_per_word();
    }

    constexpr packed_bool_container_t packed_bool_negate_mask(packed_bool_container_t mask)
    {
        return ~mask;
    }

} // namespace rocgraph
