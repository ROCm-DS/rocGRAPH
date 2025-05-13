// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*
 * Copyright (C) 2022-2024, NVIDIA CORPORATION.
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

#include "edge_property.hpp"
#include "utilities/atomic_ops_device.hpp"
#include "utilities/device_properties.hpp"
#include "utilities/packed_bool_utils.hpp"
#include "utilities/thrust_tuple_utils.hpp"

#include <thrust/iterator/iterator_traits.h>
#include <thrust/optional.h>

namespace rocgraph
{

    namespace detail
    {

        template <typename edge_t,
                  typename ValueIterator,
                  typename value_t = typename thrust::iterator_traits<ValueIterator>::value_type>
        class edge_partition_edge_property_device_view_t
        {
        public:
            using edge_type  = edge_t;
            using value_type = value_t;

            static constexpr bool is_packed_bool
                = rocgraph::is_packed_bool<ValueIterator, value_t>();
            static constexpr bool has_packed_bool_element
                = rocgraph::has_packed_bool_element<ValueIterator, value_t>();

            static_assert(
                std::is_same_v<typename thrust::iterator_traits<ValueIterator>::value_type, value_t>
                || has_packed_bool_element);
            static_assert(rocgraph::is_arithmetic_or_thrust_tuple_of_arithmetic<value_t>::value);

            edge_partition_edge_property_device_view_t() = default;

            edge_partition_edge_property_device_view_t(
                edge_property_view_t<edge_t, ValueIterator, value_t> const& view,
                size_t                                                      partition_idx)
                : value_first_(view.value_firsts()[partition_idx])
            {
                value_first_ = view.value_firsts()[partition_idx];
            }

            __host__ __device__ ValueIterator value_first() const
            {
                return value_first_;
            }

            __device__ value_t get(edge_t offset) const
            {
                if constexpr(has_packed_bool_element)
                {
                    static_assert(is_packed_bool, "unimplemented for thrust::tuple types.");

                    auto mask = rocgraph::packed_bool_mask(offset);
                    return static_cast<bool>(*(value_first_ + rocgraph::packed_bool_offset(offset))
                                             & mask);
                }
                else
                {
                    return *(value_first_ + offset);
                }
            }

            template <typename Iter = ValueIterator>
            __device__ std::enable_if_t<!std::is_const_v<std::remove_reference_t<
                                            typename std::iterator_traits<Iter>::reference>>,
                                        void>
                       set(edge_t offset, value_t val) const
            {
                if constexpr(has_packed_bool_element)
                {
                    static_assert(is_packed_bool, "unimplemented for thrust::tuple types.");

                    auto mask = rocgraph::packed_bool_mask(offset);
                    if(val)
                    {
                        atomicOr(value_first_ + rocgraph::packed_bool_offset(offset), mask);
                    }
                    else
                    {
                        atomicAnd(value_first_ + rocgraph::packed_bool_offset(offset),
                                  rocgraph::packed_bool_negate_mask(mask));
                    }
                }
                else
                {
                    *(value_first_ + offset) = val;
                }
            }

            template <typename Iter = ValueIterator>
            __device__ std::enable_if_t<!std::is_const_v<std::remove_reference_t<
                                            typename std::iterator_traits<Iter>::reference>>,
                                        value_t>
                       atomic_and(edge_t offset, value_t val) const
            {
                if constexpr(has_packed_bool_element)
                {
                    static_assert(is_packed_bool, "unimplemented for thrust::tuple types.");

                    auto mask = rocgraph::packed_bool_mask(offset);
                    auto old  = atomicAnd(value_first_ + rocgraph::packed_bool_offset(offset),
                                         val ? rocgraph::packed_bool_full_mask()
                                              : rocgraph::packed_bool_negate_mask(mask));
                    return static_cast<bool>(old & mask);
                }
                else
                {
                    return rocgraph::atomic_and(value_first_ + offset, val);
                }
            }

            template <typename Iter = ValueIterator>
            __device__ std::enable_if_t<!std::is_const_v<std::remove_reference_t<
                                            typename std::iterator_traits<Iter>::reference>>,
                                        value_t>
                       atomic_or(edge_t offset, value_t val) const
            {
                if constexpr(has_packed_bool_element)
                {
                    static_assert(is_packed_bool, "unimplemented for thrust::tuple types.");

                    auto mask = rocgraph::packed_bool_mask(offset);
                    auto old  = atomicOr(value_first_ + rocgraph::packed_bool_offset(offset),
                                        val ? mask : rocgraph::packed_bool_empty_mask());
                    return static_cast<bool>(old & mask);
                }
                else
                {
                    return rocgraph::atomic_or(value_first_ + offset, val);
                }
            }

            template <typename Iter = ValueIterator, typename T = value_t>
            __device__ std::enable_if_t<
                !std::is_const_v<
                    std::remove_reference_t<typename std::iterator_traits<Iter>::reference>>
                    && !rocgraph::
                           has_packed_bool_element<Iter, T>() /* add undefined for (packed-)bool */,
                value_t>
                atomic_add(edge_t offset, value_t val) const
            {
                rocgraph::atomic_add(value_first_ + offset, val);
            }

            template <typename Iter = ValueIterator>
            __device__ std::enable_if_t<!std::is_const_v<std::remove_reference_t<
                                            typename std::iterator_traits<Iter>::reference>>,
                                        value_t>
                       elementwise_atomic_cas(edge_t offset, value_t compare, value_t val) const
            {
                if constexpr(has_packed_bool_element)
                {
                    static_assert(is_packed_bool, "unimplemented for thrust::tuple types.");

                    auto mask = rocgraph::packed_bool_mask(offset);
                    auto ptr
                        = std::addressof(*(value_first_ + rocgraph::packed_bool_offset(offset)));
                    decltype(*ptr)    current;
                    decltype(current) target_val;
                    do
                    {
                        current
                            = __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
                        if(val)
                        {
                            target_val = current | mask;
                        }
                        else
                        {
                            target_val = current & rocgraph::packed_bool_negate_mask(mask);
                        }
                    } while(!__hip_atomic_compare_exchange_weak(
                        ptr, &current, target_val, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT));
                    return static_cast<bool>(current & mask);
                }
                else
                {
                    return rocgraph::elementwise_atomic_cas(value_first_ + offset, compare, val);
                }
            }

            template <typename Iter = ValueIterator, typename T = value_t>
            __device__ std::enable_if_t<
                !std::is_const_v<
                    std::remove_reference_t<typename std::iterator_traits<Iter>::reference>>
                    && !rocgraph::
                           has_packed_bool_element<Iter, T>() /* min undefined for (packed-)bool */,
                value_t>
                elementwise_atomic_min(edge_t offset, value_t val) const
            {
                rocgraph::elementwise_atomic_min(value_first_ + offset, val);
            }

            template <typename Iter = ValueIterator, typename T = value_t>
            __device__ std::enable_if_t<
                !std::is_const_v<
                    std::remove_reference_t<typename std::iterator_traits<Iter>::reference>>
                    && !rocgraph::
                           has_packed_bool_element<Iter, T>() /* max undefined for (packed-)bool */,
                value_t>
                elementwise_atomic_max(edge_t offset, value_t val) const
            {
                rocgraph::elementwise_atomic_max(value_first_ + offset, val);
            }

        private:
            ValueIterator value_first_{};
        };

        template <typename edge_t>
        class edge_partition_edge_dummy_property_device_view_t
        {
        public:
            using edge_type  = edge_t;
            using value_type = thrust::nullopt_t;

            static constexpr bool is_packed_bool          = false;
            static constexpr bool has_packed_bool_element = false;

            edge_partition_edge_dummy_property_device_view_t() = default;

            edge_partition_edge_dummy_property_device_view_t(edge_dummy_property_view_t const& view,
                                                             size_t partition_idx)
            {
            }

            __device__ auto get(edge_t offset) const
            {
                return thrust::nullopt;
            }
        };

    } // namespace detail

} // namespace rocgraph
