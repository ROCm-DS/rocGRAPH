// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*
 * Copyright (C) 2021-2024, NVIDIA CORPORATION.
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

#include "edge_src_dst_property.hpp"
#include "utilities/atomic_ops_device.hpp"
#include "utilities/device_functors_device.hpp"
#include "utilities/packed_bool_utils.hpp"

#include <raft/core/device_span.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/optional.h>

namespace rocgraph
{

    namespace detail
    {

        template <typename vertex_t,
                  typename ValueIterator,
                  typename value_t = typename thrust::iterator_traits<ValueIterator>::value_type>
        class edge_partition_endpoint_property_device_view_t
        {
        public:
            using vertex_type = vertex_t;
            using value_type  = value_t;
            using int_type    = typename thrust::iterator_traits<ValueIterator>::value_type;
            static constexpr bool is_packed_bool
                = rocgraph::is_packed_bool<ValueIterator, value_t>();
            static constexpr bool has_packed_bool_element
                = rocgraph::has_packed_bool_element<ValueIterator, value_t>();

            static_assert(
                std::is_same_v<typename thrust::iterator_traits<ValueIterator>::value_type, value_t>
                || has_packed_bool_element);

            edge_partition_endpoint_property_device_view_t() = default;

            edge_partition_endpoint_property_device_view_t(
                edge_major_property_view_t<vertex_t, ValueIterator, value_t> const& view,
                size_t                                                              partition_idx)
                : value_first_(view.value_firsts()[partition_idx])
                , range_first_(view.major_range_firsts()[partition_idx])
            {
                if(view.keys())
                {
                    keys_                    = (*(view.keys()))[partition_idx];
                    key_chunk_start_offsets_ = (*(view.key_chunk_start_offsets()))[partition_idx];
                    key_chunk_size_          = *(view.key_chunk_size());
                }
                value_first_ = view.value_firsts()[partition_idx];
                range_first_ = view.major_range_firsts()[partition_idx];
            }

            edge_partition_endpoint_property_device_view_t(
                edge_minor_property_view_t<vertex_t, ValueIterator, value_t> const& view)
            {
                if(view.keys())
                {
                    keys_                    = *(view.keys());
                    key_chunk_start_offsets_ = *(view.key_chunk_start_offsets());
                    key_chunk_size_          = *(view.key_chunk_size());
                }
                value_first_ = view.value_first();
                range_first_ = view.minor_range_first();
            }

            __device__ value_t get(vertex_t offset) const
            {
                auto val_offset = value_offset(offset);
                if constexpr(has_packed_bool_element)
                {
                    static_assert(is_packed_bool, "unimplemented for thrust::tuple types.");

                    auto mask = rocgraph::packed_bool_mask(val_offset);
                    return static_cast<bool>(
                        *(value_first_ + rocgraph::packed_bool_offset(val_offset)) & mask);
                }
                else
                {
                    return *(value_first_ + val_offset);
                }
            }

            template <typename Iter = ValueIterator>
            __device__ std::enable_if_t<!std::is_const_v<std::remove_reference_t<
                                            typename std::iterator_traits<Iter>::reference>>,
                                        value_t>
                       atomic_and(vertex_t offset, value_t val) const
            {
                auto val_offset = value_offset(offset);
                if constexpr(has_packed_bool_element)
                {
                    static_assert(is_packed_bool, "unimplemented for thrust::tuple types.");

                    auto mask = rocgraph::packed_bool_mask(val_offset);
                    auto old  = atomicAnd(value_first_ + rocgraph::packed_bool_offset(val_offset),
                                         val ? rocgraph::packed_bool_full_mask
                                              : rocgraph::packed_bool_negate_mask(mask));
                    return static_cast<bool>(old & mask);
                }
                else
                {
                    return rocgraph::atomic_and(value_first_ + val_offset, val);
                }
            }

            template <typename Iter = ValueIterator>
            __device__ std::enable_if_t<!std::is_const_v<std::remove_reference_t<
                                            typename std::iterator_traits<Iter>::reference>>,
                                        value_t>
                       atomic_or(vertex_t offset, value_t val) const
            {
                auto val_offset = value_offset(offset);
                if constexpr(has_packed_bool_element)
                {
                    static_assert(is_packed_bool, "unimplemented for thrust::tuple types.");

                    auto mask = rocgraph::packed_bool_mask(val_offset);
                    auto old  = atomicOr(value_first_ + rocgraph::packed_bool_offset(val_offset),
                                        val ? mask : rocgraph::packed_bool_empty_mask());
                    return static_cast<bool>(old & mask);
                }
                else
                {
                    return rocgraph::atomic_or(value_first_ + val_offset, val);
                }
            }

            template <typename Iter = ValueIterator, typename T = value_t>
            __device__ std::enable_if_t<
                !std::is_const_v<
                    std::remove_reference_t<typename std::iterator_traits<Iter>::reference>>
                && !rocgraph::has_packed_bool_element<
                    Iter,
                    T>() /* add undefined for (packed-)bool */> /* removed value_t*/
                atomic_add(vertex_t offset, value_t val) const
            {
                auto val_offset = value_offset(offset);
                rocgraph::atomic_add(value_first_ + val_offset, val);
            }

            template <typename Iter = ValueIterator>
            __device__ std::enable_if_t<!std::is_const_v<std::remove_reference_t<
                                            typename std::iterator_traits<Iter>::reference>>,
                                        value_t>
                       elementwise_atomic_cas(vertex_t offset, value_t compare, value_t val) const
            {
                auto val_offset = value_offset(offset);
                if constexpr(has_packed_bool_element)
                {
                    static_assert(is_packed_bool, "unimplemented for thrust::tuple types.");

                    auto mask = rocgraph::packed_bool_mask(val_offset);
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
                    return rocgraph::elementwise_atomic_cas(
                        value_first_ + val_offset, compare, val);
                }
            }

            template <typename Iter = ValueIterator, typename T = value_t>
            __device__ std::enable_if_t<
                !std::is_const_v<
                    std::remove_reference_t<typename std::iterator_traits<Iter>::reference>>
                && !rocgraph::has_packed_bool_element<
                    Iter,
                    T>() /* min undefined for (packed-)bool */> /* removed value_t*/
                elementwise_atomic_min(vertex_t offset, value_t val) const
            {
                auto val_offset = value_offset(offset);
                rocgraph::elementwise_atomic_min(value_first_ + val_offset, val);
            }

            template <typename Iter = ValueIterator, typename T = value_t>
            __device__ std::enable_if_t<
                !std::is_const_v<
                    std::remove_reference_t<typename std::iterator_traits<Iter>::reference>>
                && !rocgraph::has_packed_bool_element<Iter,
                                                      T>() /* max undefined for (packed-)bool */
                > /* removed value_t*/
                elementwise_atomic_max(vertex_t offset, value_t val) const
            {
                auto val_offset = value_offset(offset);
                rocgraph::elementwise_atomic_max(value_first_ + val_offset, val);
            }

        private:
            thrust::optional<raft::device_span<vertex_t const>> keys_{thrust::nullopt};
            thrust::optional<raft::device_span<vertex_t const>> key_chunk_start_offsets_{
                thrust::nullopt};
            thrust::optional<size_t> key_chunk_size_{thrust::nullopt};

            ValueIterator value_first_{};
            vertex_t      range_first_{};

            __device__ vertex_t value_offset(vertex_t offset) const
            {
                auto val_offset = offset;
                if(keys_)
                {
                    auto chunk_idx = static_cast<size_t>(offset) / (*key_chunk_size_);
                    auto it        = thrust::lower_bound(
                        thrust::seq,
                        (*keys_).begin() + (*key_chunk_start_offsets_)[chunk_idx],
                        (*keys_).begin() + (*key_chunk_start_offsets_)[chunk_idx + 1],
                        range_first_ + offset);
                    assert((it != (*keys_).begin() + (*key_chunk_start_offsets_)[chunk_idx + 1])
                           && (*it == (range_first_ + offset)));
                    val_offset
                        = (*key_chunk_start_offsets_)[chunk_idx]
                          + static_cast<vertex_t>(thrust::distance(
                              (*keys_).begin() + (*key_chunk_start_offsets_)[chunk_idx], it));
                }
                return val_offset;
            }
        };

        template <typename vertex_t>
        class edge_partition_endpoint_dummy_property_device_view_t
        {
        public:
            using vertex_type                             = vertex_t;
            using value_type                              = thrust::nullopt_t;
            static constexpr bool is_packed_bool          = false;
            static constexpr bool has_packed_bool_element = false;

            edge_partition_endpoint_dummy_property_device_view_t() = default;

            edge_partition_endpoint_dummy_property_device_view_t(
                edge_endpoint_dummy_property_view_t const& view, size_t partition_idx)
            {
            }

            edge_partition_endpoint_dummy_property_device_view_t(
                edge_endpoint_dummy_property_view_t const& view)
            {
            }

            __device__ auto get(vertex_t offset) const
            {
                return thrust::nullopt;
            }
        };

    } // namespace detail

} // namespace rocgraph
