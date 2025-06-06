// Copyright (C) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>
#include <thrust/optional.h>
#include <thrust/remove.h>
#include <thrust/tuple.h>

namespace rocgraph
{
    namespace detail
    {

        template <typename vertex_t, typename label_t>
        std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<label_t>>>
            remove_visited_vertices_from_frontier(
                raft::handle_t const&                           handle,
                rmm::device_uvector<vertex_t>&&                 frontier_vertices,
                std::optional<rmm::device_uvector<label_t>>&&   frontier_vertex_labels,
                raft::device_span<vertex_t const>               vertices_used_as_source,
                std::optional<raft::device_span<label_t const>> vertex_labels_used_as_source)
        {
            if(frontier_vertex_labels)
            {
                auto begin_iter = thrust::make_zip_iterator(frontier_vertices.begin(),
                                                            frontier_vertex_labels->begin());
                auto new_end    = thrust::remove_if(
                    handle.get_thrust_policy(),
                    begin_iter,
                    begin_iter + frontier_vertices.size(),
                    begin_iter,
                    [a_begin = vertices_used_as_source.begin(),
                     a_end   = vertices_used_as_source.end(),
                     b_begin = vertex_labels_used_as_source->begin(),
                     b_end   = vertex_labels_used_as_source
                                 ->end()] __device__(thrust::tuple<vertex_t, label_t> tuple) {
                        return thrust::binary_search(thrust::seq,
                                                     thrust::make_zip_iterator(a_begin, b_begin),
                                                     thrust::make_zip_iterator(a_end, b_end),
                                                     tuple);
                    });

                frontier_vertices.resize(thrust::distance(begin_iter, new_end),
                                         handle.get_stream());
                frontier_vertex_labels->resize(thrust::distance(begin_iter, new_end),
                                               handle.get_stream());
            }
            else
            {
                auto new_end = thrust::copy_if(
                    handle.get_thrust_policy(),
                    frontier_vertices.begin(),
                    frontier_vertices.end(),
                    frontier_vertices.begin(),
                    [a_begin = vertices_used_as_source.begin(),
                     a_end   = vertices_used_as_source.end()] __device__(vertex_t v) {
                        return !thrust::binary_search(thrust::seq, a_begin, a_end, v);
                    });
                frontier_vertices.resize(thrust::distance(frontier_vertices.begin(), new_end),
                                         handle.get_stream());
            }

            return std::make_tuple(std::move(frontier_vertices), std::move(frontier_vertex_labels));
        }

        template std::tuple<rmm::device_uvector<int32_t>,
                            std::optional<rmm::device_uvector<int32_t>>>
            remove_visited_vertices_from_frontier(
                raft::handle_t const&                           handle,
                rmm::device_uvector<int32_t>&&                  frontier_vertices,
                std::optional<rmm::device_uvector<int32_t>>&&   frontier_vertex_labels,
                raft::device_span<int32_t const>                vertices_used_as_source,
                std::optional<raft::device_span<int32_t const>> vertex_labels_used_as_source);

        template std::tuple<rmm::device_uvector<int64_t>,
                            std::optional<rmm::device_uvector<int32_t>>>
            remove_visited_vertices_from_frontier(
                raft::handle_t const&                           handle,
                rmm::device_uvector<int64_t>&&                  frontier_vertices,
                std::optional<rmm::device_uvector<int32_t>>&&   frontier_vertex_labels,
                raft::device_span<int64_t const>                vertices_used_as_source,
                std::optional<raft::device_span<int32_t const>> vertex_labels_used_as_source);

    } // namespace detail
} // namespace rocgraph
