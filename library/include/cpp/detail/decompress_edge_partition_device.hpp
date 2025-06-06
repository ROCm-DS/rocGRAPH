// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "hip/hip_runtime.h"
/*
 * Copyright (C) 2020-2024, NVIDIA CORPORATION.
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

#include "edge_partition_device_view_device.hpp"
#include "edge_partition_edge_property_device_view_device.hpp"
#include "graph_view.hpp"
#include "utilities/device_properties.hpp"
#include "utilities/mask_utils_device.hpp"
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/optional.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include "control.h"
#include <optional>
#include <tuple>
#include <vector>

namespace rocgraph
{
    namespace detail
    {

        // FIXME: block size requires tuning
        int32_t constexpr decompress_edge_partition_block_size = 1024;

        template <typename vertex_t, typename edge_t, bool multi_gpu>
        __global__ static void decompress_to_edgelist_mid_degree(
            edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition,
            vertex_t                                                  major_range_first,
            vertex_t                                                  major_range_last,
            raft::device_span<vertex_t>                               majors)
        {
            auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
            static_assert(decompress_edge_partition_block_size % raft::warp_size() == 0);
            auto const lane_id = tid % raft::warp_size();
            auto       major_start_offset
                = static_cast<size_t>(major_range_first - edge_partition.major_range_first());
            size_t idx = static_cast<size_t>(tid / raft::warp_size());

            while(idx < static_cast<size_t>(major_range_last - major_range_first))
            {
                auto major_offset = major_start_offset + idx;
                auto major        = edge_partition.major_from_major_offset_nocheck(
                    static_cast<vertex_t>(major_offset));
                vertex_t const*         indices{nullptr};
                [[maybe_unused]] edge_t edge_offset{};
                edge_t                  local_degree{};
                thrust::tie(indices, edge_offset, local_degree)
                    = edge_partition.local_edges(major_offset);
                auto local_offset = edge_partition.local_offset(major_offset);
                for(edge_t i = lane_id; i < local_degree; i += raft::warp_size())
                {
                    majors[local_offset + i] = major;
                }
                idx += gridDim.x * (blockDim.x / raft::warp_size());
            }
        }

        template <typename vertex_t, typename edge_t, bool multi_gpu>
        __global__ static void decompress_to_edgelist_high_degree(
            edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition,
            vertex_t                                                  major_range_first,
            vertex_t                                                  major_range_last,
            raft::device_span<vertex_t>                               majors)
        {
            auto major_start_offset
                = static_cast<size_t>(major_range_first - edge_partition.major_range_first());
            size_t idx = static_cast<size_t>(blockIdx.x);

            while(idx < static_cast<size_t>(major_range_last - major_range_first))
            {
                auto major_offset = major_start_offset + idx;
                auto major        = edge_partition.major_from_major_offset_nocheck(
                    static_cast<vertex_t>(major_offset));
                vertex_t const*         indices{nullptr};
                [[maybe_unused]] edge_t edge_offset{};
                edge_t                  local_degree{};
                thrust::tie(indices, edge_offset, local_degree)
                    = edge_partition.local_edges(static_cast<vertex_t>(major_offset));
                auto local_offset = edge_partition.local_offset(major_offset);
                for(edge_t i = threadIdx.x; i < local_degree; i += blockDim.x)
                {
                    majors[local_offset + i] = major;
                }
                idx += gridDim.x;
            }
        }

        template <typename vertex_t, typename edge_t, bool multi_gpu>
        void decompress_edge_partition_to_fill_edgelist_majors(
            raft::handle_t const&                                     handle,
            edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition,
            std::optional<edge_partition_edge_property_device_view_t<edge_t,
                                                                     packed_bool_container_t const*,
                                                                     bool>>
                                                        edge_partition_mask_view,
            raft::device_span<vertex_t>                 majors,
            std::optional<std::vector<vertex_t>> const& segment_offsets)
        {
            auto tmp_buffer = edge_partition_mask_view
                                  ? std::make_optional<rmm::device_uvector<vertex_t>>(
                                        edge_partition.number_of_edges(), handle.get_stream())
                                  : std::nullopt;

            auto output_buffer = tmp_buffer ? raft::device_span<vertex_t>((*tmp_buffer).data(),
                                                                          (*tmp_buffer).size())
                                            : majors;

            if(segment_offsets)
            {
                // FIXME: we may further improve performance by 1) concurrently running kernels on different
                // segments; 2) individually tuning block sizes for different segments; and 3) adding one more
                // segment for very high degree vertices and running segmented reduction
                static_assert(detail::num_sparse_segments_per_vertex_partition == 3);
                if((*segment_offsets)[1] > 0)
                {
                    raft::grid_1d_block_t update_grid((*segment_offsets)[1],
                                                      detail::decompress_edge_partition_block_size,
                                                      handle.get_device_properties().maxGridSize[0],
                                                      handle.get_device());

                    THROW_IF_HIPLAUNCHKERNELGGL_ERROR((detail::decompress_to_edgelist_high_degree),
                                                      dim3(update_grid.num_blocks),
                                                      dim3(update_grid.block_size),
                                                      0,
                                                      handle.get_stream(),
                                                      edge_partition,
                                                      edge_partition.major_range_first(),
                                                      edge_partition.major_range_first()
                                                          + (*segment_offsets)[1],
                                                      output_buffer);
                }
                if((*segment_offsets)[2] - (*segment_offsets)[1] > 0)
                {
                    raft::grid_1d_warp_t update_grid((*segment_offsets)[2] - (*segment_offsets)[1],
                                                     detail::decompress_edge_partition_block_size,
                                                     handle.get_device_properties().maxGridSize[0],
                                                     handle.get_device());

                    THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (detail::decompress_to_edgelist_mid_degree),
                        dim3(update_grid.num_blocks),
                        dim3(update_grid.block_size),
                        0,
                        handle.get_stream(),
                        edge_partition,
                        edge_partition.major_range_first() + (*segment_offsets)[1],
                        edge_partition.major_range_first() + (*segment_offsets)[2],
                        output_buffer);
                }
                if((*segment_offsets)[3] - (*segment_offsets)[2] > 0)
                {
                    thrust::for_each(
                        handle.get_thrust_policy(),
                        thrust::make_counting_iterator(edge_partition.major_range_first())
                            + (*segment_offsets)[2],
                        thrust::make_counting_iterator(edge_partition.major_range_first())
                            + (*segment_offsets)[3],
                        [edge_partition, output_buffer] __device__(auto major) {
                            auto major_offset
                                = edge_partition.major_offset_from_major_nocheck(major);
                            auto local_degree = edge_partition.local_degree(major_offset);
                            auto local_offset = edge_partition.local_offset(major_offset);
                            thrust::fill(thrust::seq,
                                         output_buffer.begin() + local_offset,
                                         output_buffer.begin() + local_offset + local_degree,
                                         major);
                        });
                }
                if(edge_partition.dcs_nzd_vertex_count()
                   && (*(edge_partition.dcs_nzd_vertex_count()) > 0))
                {
                    thrust::for_each(
                        handle.get_thrust_policy(),
                        thrust::make_counting_iterator(vertex_t{0}),
                        thrust::make_counting_iterator(*(edge_partition.dcs_nzd_vertex_count())),
                        [edge_partition,
                         major_start_offset = (*segment_offsets)[3],
                         output_buffer] __device__(auto idx) {
                            auto major
                                = *(edge_partition.major_from_major_hypersparse_idx_nocheck(idx));
                            auto major_idx
                                = major_start_offset
                                  + idx; // major_offset != major_idx in the hypersparse region
                            auto local_degree = edge_partition.local_degree(major_idx);
                            auto local_offset = edge_partition.local_offset(major_idx);
                            thrust::fill(thrust::seq,
                                         output_buffer.begin() + local_offset,
                                         output_buffer.begin() + local_offset + local_degree,
                                         major);
                        });
                }
            }
            else
            {
                thrust::for_each(
                    handle.get_thrust_policy(),
                    thrust::make_counting_iterator(edge_partition.major_range_first()),
                    thrust::make_counting_iterator(edge_partition.major_range_first())
                        + edge_partition.major_range_size(),
                    [edge_partition, output_buffer] __device__(auto major) {
                        auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
                        auto local_degree = edge_partition.local_degree(major_offset);
                        auto local_offset = edge_partition.local_offset(major_offset);
                        thrust::fill(thrust::seq,
                                     output_buffer.begin() + local_offset,
                                     output_buffer.begin() + local_offset + local_degree,
                                     major);
                    });
            }

            if(tmp_buffer)
            {
                copy_if_mask_set(handle,
                                 (*tmp_buffer).begin(),
                                 (*tmp_buffer).end(),
                                 (*edge_partition_mask_view).value_first(),
                                 majors.begin());
            }
        }

        template <typename vertex_t,
                  typename edge_t,
                  typename weight_t,
                  typename edge_type_t,
                  bool multi_gpu>
        void decompress_edge_partition_to_edgelist(
            raft::handle_t const&                                     handle,
            edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition,
            std::optional<edge_partition_edge_property_device_view_t<edge_t, weight_t const*>>
                edge_partition_weight_view,
            std::optional<edge_partition_edge_property_device_view_t<edge_t, edge_t const*>>
                edge_partition_id_view,
            std::optional<edge_partition_edge_property_device_view_t<edge_t, edge_type_t const*>>
                edge_partition_type_view,
            std::optional<edge_partition_edge_property_device_view_t<edge_t,
                                                                     packed_bool_container_t const*,
                                                                     bool>>
                                                          edge_partition_mask_view,
            raft::device_span<vertex_t>                   edgelist_majors /* [OUT] */,
            raft::device_span<vertex_t>                   edgelist_minors /* [OUT] */,
            std::optional<raft::device_span<weight_t>>    edgelist_weights /* [OUT] */,
            std::optional<raft::device_span<edge_t>>      edgelist_ids /* [OUT] */,
            std::optional<raft::device_span<edge_type_t>> edgelist_types /* [OUT] */,
            std::optional<std::vector<vertex_t>> const&   segment_offsets)
        {
            auto number_of_edges = edge_partition.number_of_edges();

            decompress_edge_partition_to_fill_edgelist_majors(
                handle, edge_partition, edge_partition_mask_view, edgelist_majors, segment_offsets);
            if(edge_partition_mask_view)
            {
                copy_if_mask_set(handle,
                                 edge_partition.indices(),
                                 edge_partition.indices() + number_of_edges,
                                 (*edge_partition_mask_view).value_first(),
                                 edgelist_minors.begin());
            }
            else
            {
                thrust::copy(handle.get_thrust_policy(),
                             edge_partition.indices(),
                             edge_partition.indices() + number_of_edges,
                             edgelist_minors.begin());
            }
            if(edge_partition_weight_view)
            {
                assert(edgelist_weights.has_value());
                if(edge_partition_mask_view)
                {
                    copy_if_mask_set(handle,
                                     (*edge_partition_weight_view).value_first(),
                                     (*edge_partition_weight_view).value_first() + number_of_edges,
                                     (*edge_partition_mask_view).value_first(),
                                     (*edgelist_weights).begin());
                }
                else
                {
                    thrust::copy(handle.get_thrust_policy(),
                                 (*edge_partition_weight_view).value_first(),
                                 (*edge_partition_weight_view).value_first() + number_of_edges,
                                 (*edgelist_weights).begin());
                }
            }
            if(edge_partition_id_view)
            {
                assert(edgelist_ids.has_value());
                if(edge_partition_mask_view)
                {
                    copy_if_mask_set(handle,
                                     (*edge_partition_id_view).value_first(),
                                     (*edge_partition_id_view).value_first() + number_of_edges,
                                     (*edge_partition_mask_view).value_first(),
                                     (*edgelist_ids).begin());
                }
                else
                {
                    thrust::copy(handle.get_thrust_policy(),
                                 (*edge_partition_id_view).value_first(),
                                 (*edge_partition_id_view).value_first() + number_of_edges,
                                 (*edgelist_ids).begin());
                }
            }

            if(edge_partition_type_view)
            {
                assert(edgelist_types.has_value());
                if(edge_partition_mask_view)
                {
                    copy_if_mask_set(handle,
                                     (*edge_partition_type_view).value_first(),
                                     (*edge_partition_type_view).value_first() + number_of_edges,
                                     (*edge_partition_mask_view).value_first(),
                                     (*edgelist_types).begin());
                }
                else
                {
                    thrust::copy(handle.get_thrust_policy(),
                                 (*edge_partition_type_view).value_first(),
                                 (*edge_partition_type_view).value_first() + number_of_edges,
                                 (*edgelist_types).begin());
                }
            }
        }

    } // namespace detail
} // namespace rocgraph
