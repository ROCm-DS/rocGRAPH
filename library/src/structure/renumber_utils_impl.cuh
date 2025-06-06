// Copyright (C) 2020-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "detail/graph_partition_utils.cuh"
#include "prims/kv_store.cuh"
#include "utilities/collect_comm.cuh"

#include "graph.hpp"
#include "graph_functions.hpp"
#include "utilities/error.hpp"
#include "utilities/host_scalar_comm.hpp"
#include "utilities/shuffle_comm_device.hpp"

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

namespace rocgraph
{

    namespace detail
    {

        template <typename vertex_t>
        void unrenumber_local_int_edges(raft::handle_t const&         handle,
                                        std::vector<vertex_t*> const& edgelist_majors /* [INOUT] */,
                                        std::vector<vertex_t*> const& edgelist_minors /* [INOUT] */,
                                        std::vector<size_t> const&    edgelist_edge_counts,
                                        vertex_t const*               renumber_map_labels,
                                        std::vector<vertex_t> const&  vertex_partition_range_lasts,
                                        std::optional<std::vector<std::vector<size_t>>> const&
                                             edgelist_intra_partition_segment_offsets,
                                        bool do_expensive_check)
        {
            auto&      comm      = handle.get_comms();
            auto const comm_size = comm.get_size();
            auto const comm_rank = comm.get_rank();
            auto& major_comm = handle.get_subcomm(rocgraph::partition_manager::major_comm_name());
            auto const major_comm_size = major_comm.get_size();
            auto const major_comm_rank = major_comm.get_rank();
            auto& minor_comm = handle.get_subcomm(rocgraph::partition_manager::minor_comm_name());
            auto const minor_comm_size = minor_comm.get_size();
            auto const minor_comm_rank = minor_comm.get_rank();

            ROCGRAPH_EXPECTS(edgelist_majors.size() == static_cast<size_t>(minor_comm_size),
                             "Invalid input arguments: erroneous edgelist_majors.size().");
            ROCGRAPH_EXPECTS(edgelist_minors.size() == static_cast<size_t>(minor_comm_size),
                             "Invalid input arguments: erroneous edgelist_minors.size().");
            ROCGRAPH_EXPECTS(edgelist_edge_counts.size() == static_cast<size_t>(minor_comm_size),
                             "Invalid input arguments: erroneous edgelist_edge_counts.size().");
            ROCGRAPH_EXPECTS(
                std::is_sorted(vertex_partition_range_lasts.begin(),
                               vertex_partition_range_lasts.end()),
                "Invalid input arguments: vertex_partition_range_lasts is not sorted.");
            if(edgelist_intra_partition_segment_offsets)
            {
                ROCGRAPH_EXPECTS((*edgelist_intra_partition_segment_offsets).size()
                                     == static_cast<size_t>(minor_comm_size),
                                 "Invalid input arguments: erroneous "
                                 "(*edgelist_intra_partition_segment_offsets).size().");
                for(size_t i = 0; i < edgelist_majors.size(); ++i)
                {
                    ROCGRAPH_EXPECTS((*edgelist_intra_partition_segment_offsets)[i].size()
                                         == static_cast<size_t>(major_comm_size + 1),
                                     "Invalid input arguments: erroneous "
                                     "(*edgelist_intra_partition_segment_offsets)[].size().");
                    ROCGRAPH_EXPECTS(
                        std::is_sorted((*edgelist_intra_partition_segment_offsets)[i].begin(),
                                       (*edgelist_intra_partition_segment_offsets)[i].end()),
                        "Invalid input arguments: (*edgelist_intra_partition_segment_offsets)[] is "
                        "not sorted.");
                    ROCGRAPH_EXPECTS(
                        ((*edgelist_intra_partition_segment_offsets)[i][0] == 0)
                            && ((*edgelist_intra_partition_segment_offsets)[i].back()
                                == edgelist_edge_counts[i]),
                        "Invalid input arguments: (*edgelist_intra_partition_segment_offsets)[][0] "
                        "should be 0 and "
                        "(*edgelist_intra_partition_segment_offsets)[].back() should coincide with "
                        "edgelist_edge_counts[].");
                }
            }

            if(do_expensive_check)
            {
                for(size_t i = 0; i < edgelist_majors.size(); ++i)
                {
                    auto major_range_vertex_partition_id
                        = compute_local_edge_partition_major_range_vertex_partition_id_t{
                            major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
                    auto edge_partition_major_range_first
                        = major_range_vertex_partition_id == 0
                              ? vertex_t{0}
                              : vertex_partition_range_lasts[major_range_vertex_partition_id - 1];
                    auto edge_partition_major_range_last
                        = vertex_partition_range_lasts[major_range_vertex_partition_id];
                    ROCGRAPH_EXPECTS(
                        thrust::count_if(handle.get_thrust_policy(),
                                         edgelist_majors[i],
                                         edgelist_majors[i] + edgelist_edge_counts[i],
                                         [edge_partition_major_range_first,
                                          edge_partition_major_range_last] __device__(auto v) {
                                             return v != invalid_vertex_id<vertex_t>::value
                                                    && (v < edge_partition_major_range_first
                                                        || v >= edge_partition_major_range_last);
                                         })
                            == 0,
                        "Invalid input arguments: there are out-of-range vertices in "
                        "[edgelist_majors[], "
                        "edgelist_majors[] + edgelist_edge_counts[]).");

                    if(edgelist_intra_partition_segment_offsets)
                    {
                        for(int j = 0; j < major_comm_size; ++j)
                        {
                            auto minor_range_vertex_partition_id
                                = compute_local_edge_partition_minor_range_vertex_partition_id_t{
                                    major_comm_size,
                                    minor_comm_size,
                                    major_comm_rank,
                                    minor_comm_rank}(j);
                            auto valid_first
                                = minor_range_vertex_partition_id == 0
                                      ? vertex_t{0}
                                      : vertex_partition_range_lasts[minor_range_vertex_partition_id
                                                                     - 1];
                            auto valid_last
                                = vertex_partition_range_lasts[minor_range_vertex_partition_id];
                            ROCGRAPH_EXPECTS(
                                thrust::count_if(
                                    handle.get_thrust_policy(),
                                    edgelist_minors[i]
                                        + (*edgelist_intra_partition_segment_offsets)[i][j],
                                    edgelist_minors[i]
                                        + (*edgelist_intra_partition_segment_offsets)[i][j + 1],
                                    [valid_first, valid_last] __device__(auto v) {
                                        return v != invalid_vertex_id<vertex_t>::value
                                               && (v < valid_first || v >= valid_last);
                                    })
                                    == 0,
                                "Invalid input arguments: there are out-of-range vertices in "
                                "[edgelist_minors[], "
                                "edgelist_minors[] + edgelist_edge_counts[]).");
                        }
                    }
                    else
                    {
                        auto minor_range_first_vertex_partition_id
                            = compute_local_edge_partition_minor_range_vertex_partition_id_t{
                                major_comm_size,
                                minor_comm_size,
                                major_comm_rank,
                                minor_comm_rank}(0);
                        auto minor_range_last_vertex_partition_id
                            = compute_local_edge_partition_minor_range_vertex_partition_id_t{
                                major_comm_size,
                                minor_comm_size,
                                major_comm_rank,
                                minor_comm_rank}(major_comm_size - 1);
                        auto edge_partition_minor_range_first
                            = minor_range_first_vertex_partition_id == 0
                                  ? vertex_t{0}
                                  : vertex_partition_range_lasts
                                        [minor_range_first_vertex_partition_id - 1];
                        auto edge_partition_minor_range_last
                            = vertex_partition_range_lasts[minor_range_last_vertex_partition_id];
                        ROCGRAPH_EXPECTS(
                            thrust::count_if(
                                handle.get_thrust_policy(),
                                edgelist_minors[i],
                                edgelist_minors[i] + edgelist_edge_counts[i],
                                [edge_partition_minor_range_first,
                                 edge_partition_minor_range_last] __device__(auto v) {
                                    return v != invalid_vertex_id<vertex_t>::value
                                           && (v < edge_partition_minor_range_first
                                               || v >= edge_partition_minor_range_last);
                                })
                                == 0,
                            "Invalid input arguments: there are out-of-range vertices in "
                            "[edgelist_minors[], "
                            "edgelist_minors[] + edgelist_edge_counts[]).");
                    }
                }
            }

            auto number_of_edges = host_scalar_allreduce(
                comm,
                std::reduce(edgelist_edge_counts.begin(), edgelist_edge_counts.end()),
                raft::comms::op_t::SUM,
                handle.get_stream());

            // FIXME: compare this hash based approach with a binary search based approach in both memory
            // footprint and execution time

            {
                vertex_t max_edge_partition_major_range_size{0};
                for(size_t i = 0; i < edgelist_majors.size(); ++i)
                {
                    auto major_range_vertex_partition_id
                        = compute_local_edge_partition_major_range_vertex_partition_id_t{
                            major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
                    auto edge_partition_major_range_size
                        = vertex_partition_range_lasts[major_range_vertex_partition_id]
                          - (major_range_vertex_partition_id == 0
                                 ? vertex_t{0}
                                 : vertex_partition_range_lasts[major_range_vertex_partition_id
                                                                - 1]);
                    max_edge_partition_major_range_size = std::max(
                        max_edge_partition_major_range_size, edge_partition_major_range_size);
                }
                rmm::device_uvector<vertex_t> renumber_map_major_labels(
                    max_edge_partition_major_range_size, handle.get_stream());
                for(size_t i = 0; i < edgelist_majors.size(); ++i)
                {
                    auto major_range_vertex_partition_id
                        = compute_local_edge_partition_major_range_vertex_partition_id_t{
                            major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
                    auto edge_partition_major_range_first
                        = major_range_vertex_partition_id == 0
                              ? vertex_t{0}
                              : vertex_partition_range_lasts[major_range_vertex_partition_id - 1];
                    auto edge_partition_major_range_last
                        = vertex_partition_range_lasts[major_range_vertex_partition_id];
                    auto edge_partition_major_range_size
                        = edge_partition_major_range_last - edge_partition_major_range_first;
                    device_bcast(minor_comm,
                                 renumber_map_labels,
                                 renumber_map_major_labels.data(),
                                 edge_partition_major_range_size,
                                 i,
                                 handle.get_stream());

                    kv_store_t<vertex_t, vertex_t, false> renumber_map(
                        thrust::make_counting_iterator(edge_partition_major_range_first),
                        thrust::make_counting_iterator(edge_partition_major_range_first)
                            + edge_partition_major_range_size,
                        renumber_map_major_labels.begin(),
                        invalid_vertex_id<vertex_t>::value,
                        invalid_vertex_id<vertex_t>::value,
                        handle.get_stream());
                    auto renumber_map_view = renumber_map.view();
                    renumber_map_view.find(edgelist_majors[i],
                                           edgelist_majors[i] + edgelist_edge_counts[i],
                                           edgelist_majors[i],
                                           handle.get_stream());
                }
            }

            vertex_t edge_partition_minor_range_size{0};
            for(int i = 0; i < major_comm_size; ++i)
            {
                auto minor_range_vertex_partition_id
                    = compute_local_edge_partition_minor_range_vertex_partition_id_t{
                        major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
                edge_partition_minor_range_size
                    += vertex_partition_range_lasts[minor_range_vertex_partition_id]
                       - (minor_range_vertex_partition_id == 0
                              ? vertex_t{0}
                              : vertex_partition_range_lasts[minor_range_vertex_partition_id - 1]);
            }
            if((edge_partition_minor_range_size
                >= static_cast<vertex_t>(number_of_edges / comm_size))
               && edgelist_intra_partition_segment_offsets)
            { // memory footprint dominated by the O(V/sqrt(P))
                // part than the O(E/P) part
                vertex_t max_segment_size{0};
                for(int i = 0; i < major_comm_size; ++i)
                {
                    auto minor_range_vertex_partition_id
                        = compute_local_edge_partition_minor_range_vertex_partition_id_t{
                            major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
                    max_segment_size = std::max(
                        max_segment_size,
                        vertex_partition_range_lasts[minor_range_vertex_partition_id]
                            - (minor_range_vertex_partition_id == 0
                                   ? vertex_t{0}
                                   : vertex_partition_range_lasts[minor_range_vertex_partition_id
                                                                  - 1]));
                }
                rmm::device_uvector<vertex_t> renumber_map_minor_labels(max_segment_size,
                                                                        handle.get_stream());
                for(int i = 0; i < major_comm_size; ++i)
                {
                    auto minor_range_vertex_partition_id
                        = compute_local_edge_partition_minor_range_vertex_partition_id_t{
                            major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
                    auto vertex_partition_minor_range_first
                        = (minor_range_vertex_partition_id == 0)
                              ? vertex_t{0}
                              : vertex_partition_range_lasts[minor_range_vertex_partition_id - 1];
                    auto vertex_partition_minor_range_last
                        = vertex_partition_range_lasts[minor_range_vertex_partition_id];
                    auto segment_size
                        = vertex_partition_minor_range_last - vertex_partition_minor_range_first;
                    device_bcast(major_comm,
                                 renumber_map_labels,
                                 renumber_map_minor_labels.data(),
                                 segment_size,
                                 i,
                                 handle.get_stream());

                    kv_store_t<vertex_t, vertex_t, false> renumber_map(
                        thrust::make_counting_iterator(vertex_partition_minor_range_first),
                        thrust::make_counting_iterator(vertex_partition_minor_range_first)
                            + segment_size,
                        renumber_map_minor_labels.begin(),
                        invalid_vertex_id<vertex_t>::value,
                        invalid_vertex_id<vertex_t>::value,
                        handle.get_stream());
                    auto renumber_map_view = renumber_map.view();
                    for(size_t j = 0; j < edgelist_minors.size(); ++j)
                    {
                        renumber_map_view.find(
                            edgelist_minors[j] + (*edgelist_intra_partition_segment_offsets)[j][i],
                            edgelist_minors[j]
                                + (*edgelist_intra_partition_segment_offsets)[j][i + 1],
                            edgelist_minors[j] + (*edgelist_intra_partition_segment_offsets)[j][i],
                            handle.get_stream());
                    }
                }
            }
            else
            {
                auto minor_range_first_vertex_partition_id
                    = compute_local_edge_partition_minor_range_vertex_partition_id_t{
                        major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(0);
                auto edge_partition_minor_range_first
                    = minor_range_first_vertex_partition_id == 0
                          ? vertex_t{0}
                          : vertex_partition_range_lasts[minor_range_first_vertex_partition_id - 1];
                rmm::device_uvector<vertex_t> renumber_map_minor_labels(
                    edge_partition_minor_range_size, handle.get_stream());
                std::vector<size_t> recvcounts(major_comm_size);
                for(int i = 0; i < major_comm_size; ++i)
                {
                    auto minor_range_vertex_partition_id
                        = compute_local_edge_partition_minor_range_vertex_partition_id_t{
                            major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
                    recvcounts[i]
                        = vertex_partition_range_lasts[minor_range_vertex_partition_id]
                          - (minor_range_vertex_partition_id == 0
                                 ? vertex_t{0}
                                 : vertex_partition_range_lasts[minor_range_vertex_partition_id
                                                                - 1]);
                }
                std::vector<size_t> displacements(recvcounts.size(), 0);
                std::partial_sum(
                    recvcounts.begin(), recvcounts.end() - 1, displacements.begin() + 1);
                device_allgatherv(major_comm,
                                  renumber_map_labels,
                                  renumber_map_minor_labels.begin(),
                                  recvcounts,
                                  displacements,
                                  handle.get_stream());

                kv_store_t<vertex_t, vertex_t, false> renumber_map(
                    thrust::make_counting_iterator(edge_partition_minor_range_first),
                    thrust::make_counting_iterator(edge_partition_minor_range_first)
                        + renumber_map_minor_labels.size(),
                    renumber_map_minor_labels.begin(),
                    invalid_vertex_id<vertex_t>::value,
                    invalid_vertex_id<vertex_t>::value,
                    handle.get_stream());
                auto renumber_map_view = renumber_map.view();
                for(size_t i = 0; i < edgelist_minors.size(); ++i)
                {
                    renumber_map_view.find(edgelist_minors[i],
                                           edgelist_minors[i] + edgelist_edge_counts[i],
                                           edgelist_minors[i],
                                           handle.get_stream());
                }
            }
        }

    } // namespace detail

    template <typename vertex_t, bool multi_gpu>
    void renumber_ext_vertices(raft::handle_t const& handle,
                               vertex_t*             vertices /* [INOUT] */,
                               size_t                num_vertices,
                               vertex_t const*       renumber_map_labels,
                               vertex_t              local_int_vertex_first,
                               vertex_t              local_int_vertex_last,
                               bool                  do_expensive_check)
    {
        if(do_expensive_check)
        {
            rmm::device_uvector<vertex_t> labels(local_int_vertex_last - local_int_vertex_first,
                                                 handle.get_stream());
            thrust::copy(handle.get_thrust_policy(),
                         renumber_map_labels,
                         renumber_map_labels + labels.size(),
                         labels.begin());
            thrust::sort(handle.get_thrust_policy(), labels.begin(), labels.end());
            ROCGRAPH_EXPECTS(
                thrust::unique(handle.get_thrust_policy(), labels.begin(), labels.end())
                    == labels.end(),
                "Invalid input arguments: renumber_map_labels have duplicate elements.");
        }

        std::unique_ptr<kv_store_t<vertex_t, vertex_t, false>> renumber_map_ptr{nullptr};
        if(multi_gpu)
        {
            auto&      comm      = handle.get_comms();
            auto const comm_size = comm.get_size();
            auto& major_comm = handle.get_subcomm(rocgraph::partition_manager::major_comm_name());
            auto const major_comm_size = major_comm.get_size();
            auto& minor_comm = handle.get_subcomm(rocgraph::partition_manager::minor_comm_name());
            auto const minor_comm_size = minor_comm.get_size();

            rmm::device_uvector<vertex_t> sorted_unique_ext_vertices(num_vertices,
                                                                     handle.get_stream());
            sorted_unique_ext_vertices.resize(
                thrust::distance(sorted_unique_ext_vertices.begin(),
                                 thrust::copy_if(handle.get_thrust_policy(),
                                                 vertices,
                                                 vertices + num_vertices,
                                                 sorted_unique_ext_vertices.begin(),
                                                 [] __device__(auto v) {
                                                     return v != invalid_vertex_id<vertex_t>::value;
                                                 })),
                handle.get_stream());
            thrust::sort(handle.get_thrust_policy(),
                         sorted_unique_ext_vertices.begin(),
                         sorted_unique_ext_vertices.end());
            sorted_unique_ext_vertices.resize(
                thrust::distance(sorted_unique_ext_vertices.begin(),
                                 thrust::unique(handle.get_thrust_policy(),
                                                sorted_unique_ext_vertices.begin(),
                                                sorted_unique_ext_vertices.end())),
                handle.get_stream());

            kv_store_t<vertex_t, vertex_t, false> local_renumber_map(
                renumber_map_labels,
                renumber_map_labels + (local_int_vertex_last - local_int_vertex_first),
                thrust::make_counting_iterator(local_int_vertex_first),
                invalid_vertex_id<vertex_t>::value,
                invalid_vertex_id<vertex_t>::value,
                handle.get_stream()); // map local external vertex IDs to local internal vertex IDs

            rmm::device_uvector<vertex_t> int_vertices_for_sorted_unique_ext_vertices(
                0, handle.get_stream());
            auto [unique_ext_vertices, int_vertices_for_unique_ext_vertices]
                = collect_values_for_unique_keys(handle,
                                                 local_renumber_map.view(),
                                                 std::move(sorted_unique_ext_vertices),
                                                 detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{
                                                     comm_size, major_comm_size, minor_comm_size});

            renumber_map_ptr = std::make_unique<kv_store_t<vertex_t, vertex_t, false>>(
                unique_ext_vertices.begin(),
                unique_ext_vertices.begin() + unique_ext_vertices.size(),
                int_vertices_for_unique_ext_vertices.begin(),
                invalid_vertex_id<vertex_t>::value,
                invalid_vertex_id<vertex_t>::value,
                handle.get_stream());
        }
        else
        {
            renumber_map_ptr = std::make_unique<kv_store_t<vertex_t, vertex_t, false>>(
                renumber_map_labels,
                renumber_map_labels + (local_int_vertex_last - local_int_vertex_first),
                thrust::make_counting_iterator(vertex_t{0}),
                invalid_vertex_id<vertex_t>::value,
                invalid_vertex_id<vertex_t>::value,
                handle.get_stream());
        }
        auto renumber_map_view = renumber_map_ptr->view();

        if(do_expensive_check)
        {
            rmm::device_uvector<bool> contains(num_vertices, handle.get_stream());
            renumber_map_view.contains(
                vertices, vertices + num_vertices, contains.begin(), handle.get_stream());
            auto vc_pair_first
                = thrust::make_zip_iterator(thrust::make_tuple(vertices, contains.begin()));
            ROCGRAPH_EXPECTS(thrust::count_if(handle.get_thrust_policy(),
                                              vc_pair_first,
                                              vc_pair_first + num_vertices,
                                              [] __device__(auto pair) {
                                                  auto v = thrust::get<0>(pair);
                                                  auto c = thrust::get<1>(pair);
                                                  return v == invalid_vertex_id<vertex_t>::value
                                                             ? (c == true)
                                                             : (c == false);
                                              })
                                 == 0,
                             "Invalid input arguments: vertices have elements that are missing in "
                             "(aggregate) renumber_map_labels.");
        }

        renumber_map_view.find(vertices, vertices + num_vertices, vertices, handle.get_stream());
    }

    template <typename vertex_t, bool multi_gpu>
    void renumber_local_ext_vertices(raft::handle_t const& handle,
                                     vertex_t*             vertices /* [INOUT] */,
                                     size_t                num_vertices,
                                     vertex_t const*       renumber_map_labels,
                                     vertex_t              local_int_vertex_first,
                                     vertex_t              local_int_vertex_last,
                                     bool                  do_expensive_check)
    {
        if(do_expensive_check)
        {
            rmm::device_uvector<vertex_t> labels(local_int_vertex_last - local_int_vertex_first,
                                                 handle.get_stream());
            thrust::copy(handle.get_thrust_policy(),
                         renumber_map_labels,
                         renumber_map_labels + labels.size(),
                         labels.begin());
            thrust::sort(handle.get_thrust_policy(), labels.begin(), labels.end());
            ROCGRAPH_EXPECTS(
                thrust::unique(handle.get_thrust_policy(), labels.begin(), labels.end())
                    == labels.end(),
                "Invalid input arguments: renumber_map_labels have duplicate elements.");
        }

        kv_store_t<vertex_t, vertex_t, false> renumber_map(
            renumber_map_labels,
            renumber_map_labels + (local_int_vertex_last - local_int_vertex_first),
            thrust::make_counting_iterator(local_int_vertex_first),
            invalid_vertex_id<vertex_t>::value,
            invalid_vertex_id<vertex_t>::value,
            handle.get_stream());
        auto renumber_map_view = renumber_map.view();

        if(do_expensive_check)
        {
            rmm::device_uvector<bool> contains(num_vertices, handle.get_stream());
            renumber_map_view.contains(
                vertices, vertices + num_vertices, contains.begin(), handle.get_stream());
            auto vc_pair_first
                = thrust::make_zip_iterator(thrust::make_tuple(vertices, contains.begin()));
            ROCGRAPH_EXPECTS(thrust::count_if(handle.get_thrust_policy(),
                                              vc_pair_first,
                                              vc_pair_first + num_vertices,
                                              [] __device__(auto pair) {
                                                  auto v = thrust::get<0>(pair);
                                                  auto c = thrust::get<1>(pair);
                                                  return v == invalid_vertex_id<vertex_t>::value
                                                             ? (c == true)
                                                             : (c == false);
                                              })
                                 == 0,
                             "Invalid input arguments: vertices have elements that are missing in "
                             "(aggregate) renumber_map_labels.");
        }

        renumber_map_view.find(vertices, vertices + num_vertices, vertices, handle.get_stream());
    }

    template <typename vertex_t>
    void unrenumber_local_int_vertices(
        raft::handle_t const& handle,
        vertex_t*             vertices /* [INOUT] */,
        size_t                num_vertices,
        vertex_t const*
                 renumber_map_labels /* size = local_int_vertex_last - local_int_vertex_first */,
        vertex_t local_int_vertex_first,
        vertex_t local_int_vertex_last,
        bool     do_expensive_check)
    {
        if(do_expensive_check)
        {
            ROCGRAPH_EXPECTS(
                thrust::count_if(
                    handle.get_thrust_policy(),
                    vertices,
                    vertices + num_vertices,
                    [local_int_vertex_first, local_int_vertex_last] __device__(auto v) {
                        return v != invalid_vertex_id<vertex_t>::value
                               && (v < local_int_vertex_first || v >= local_int_vertex_last);
                    })
                    == 0,
                "Invalid input arguments: there are non-local vertices in [vertices, vertices "
                "+ num_vertices).");
        }

        thrust::transform(handle.get_thrust_policy(),
                          vertices,
                          vertices + num_vertices,
                          vertices,
                          [renumber_map_labels, local_int_vertex_first] __device__(auto v) {
                              return v == invalid_vertex_id<vertex_t>::value
                                         ? v
                                         : renumber_map_labels[v - local_int_vertex_first];
                          });
    }

    template <typename vertex_t, bool multi_gpu>
    void unrenumber_int_vertices(raft::handle_t const&        handle,
                                 vertex_t*                    vertices /* [INOUT] */,
                                 size_t                       num_vertices,
                                 vertex_t const*              renumber_map_labels,
                                 std::vector<vertex_t> const& vertex_partition_range_lasts,
                                 bool                         do_expensive_check)
    {
        if(do_expensive_check)
        {
            ROCGRAPH_EXPECTS(
                thrust::count_if(
                    handle.get_thrust_policy(),
                    vertices,
                    vertices + num_vertices,
                    [int_vertex_last = vertex_partition_range_lasts.back()] __device__(auto v) {
                        return v != invalid_vertex_id<vertex_t>::value
                               && !is_valid_vertex(int_vertex_last, v);
                    })
                    == 0,
                "Invalid input arguments: there are out-of-range vertices in [vertices, vertices "
                "+ num_vertices).");
        }

        if(multi_gpu)
        {
            auto&      comm      = handle.get_comms();
            auto const comm_size = comm.get_size();
            auto const comm_rank = comm.get_rank();
            auto& major_comm = handle.get_subcomm(rocgraph::partition_manager::major_comm_name());
            auto const major_comm_size = major_comm.get_size();
            auto const major_comm_rank = major_comm.get_rank();
            auto& minor_comm = handle.get_subcomm(rocgraph::partition_manager::minor_comm_name());
            auto const minor_comm_size = minor_comm.get_size();
            auto const minor_comm_rank = minor_comm.get_rank();

            auto vertex_partition_id
                = partition_manager::compute_vertex_partition_id_from_graph_subcomm_ranks(
                    major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank);
            auto local_int_vertex_first
                = vertex_partition_id == 0 ? vertex_t{0}
                                           : vertex_partition_range_lasts[vertex_partition_id - 1];
            auto local_int_vertex_last = vertex_partition_range_lasts[vertex_partition_id];

            rmm::device_uvector<vertex_t> sorted_unique_int_vertices(num_vertices,
                                                                     handle.get_stream());
            sorted_unique_int_vertices.resize(
                thrust::distance(sorted_unique_int_vertices.begin(),
                                 thrust::copy_if(handle.get_thrust_policy(),
                                                 vertices,
                                                 vertices + num_vertices,
                                                 sorted_unique_int_vertices.begin(),
                                                 [] __device__(auto v) {
                                                     return v != invalid_vertex_id<vertex_t>::value;
                                                 })),
                handle.get_stream());
            thrust::sort(handle.get_thrust_policy(),
                         sorted_unique_int_vertices.begin(),
                         sorted_unique_int_vertices.end());
            sorted_unique_int_vertices.resize(
                thrust::distance(sorted_unique_int_vertices.begin(),
                                 thrust::unique(handle.get_thrust_policy(),
                                                sorted_unique_int_vertices.begin(),
                                                sorted_unique_int_vertices.end())),
                handle.get_stream());

            auto [unique_int_vertices, ext_vertices_for_unique_int_vertices]
                = collect_values_for_unique_int_vertices(handle,
                                                         std::move(sorted_unique_int_vertices),
                                                         renumber_map_labels,
                                                         vertex_partition_range_lasts);

            kv_store_t<vertex_t, vertex_t, false> renumber_map(
                unique_int_vertices.begin(),
                unique_int_vertices.begin() + unique_int_vertices.size(),
                ext_vertices_for_unique_int_vertices.begin(),
                invalid_vertex_id<vertex_t>::value,
                invalid_vertex_id<vertex_t>::value,
                handle.get_stream());
            auto renumber_map_view = renumber_map.view();
            renumber_map_view.find(
                vertices, vertices + num_vertices, vertices, handle.get_stream());
        }
        else
        {
            unrenumber_local_int_vertices(handle,
                                          vertices,
                                          num_vertices,
                                          renumber_map_labels,
                                          vertex_t{0},
                                          vertex_partition_range_lasts[0],
                                          do_expensive_check);
        }
    }

    template <typename vertex_t, bool store_transposed, bool multi_gpu>
    std::enable_if_t<multi_gpu, void>
        unrenumber_local_int_edges(raft::handle_t const&         handle,
                                   std::vector<vertex_t*> const& edgelist_srcs /* [INOUT] */,
                                   std::vector<vertex_t*> const& edgelist_dsts /* [INOUT] */,
                                   std::vector<size_t> const&    edgelist_edge_counts,
                                   vertex_t const*               renumber_map_labels,
                                   std::vector<vertex_t> const&  vertex_partition_range_lasts,
                                   std::optional<std::vector<std::vector<size_t>>> const&
                                        edgelist_intra_partition_segment_offsets,
                                   bool do_expensive_check)
    {
        return detail::unrenumber_local_int_edges(handle,
                                                  store_transposed ? edgelist_dsts : edgelist_srcs,
                                                  store_transposed ? edgelist_srcs : edgelist_dsts,
                                                  edgelist_edge_counts,
                                                  renumber_map_labels,
                                                  vertex_partition_range_lasts,
                                                  edgelist_intra_partition_segment_offsets,
                                                  do_expensive_check);
    }

    template <typename vertex_t, bool store_transposed, bool multi_gpu>
    std::enable_if_t<!multi_gpu, void>
        unrenumber_local_int_edges(raft::handle_t const& handle,
                                   vertex_t*             edgelist_srcs /* [INOUT] */,
                                   vertex_t*             edgelist_dsts /* [INOUT] */,
                                   size_t                num_edgelist_edges,
                                   vertex_t const*       renumber_map_labels,
                                   vertex_t              num_vertices,
                                   bool                  do_expensive_check)
    {
        unrenumber_local_int_vertices(handle,
                                      edgelist_srcs,
                                      num_edgelist_edges,
                                      renumber_map_labels,
                                      vertex_t{0},
                                      num_vertices,
                                      do_expensive_check);
        unrenumber_local_int_vertices(handle,
                                      edgelist_dsts,
                                      num_edgelist_edges,
                                      renumber_map_labels,
                                      vertex_t{0},
                                      num_vertices,
                                      do_expensive_check);
    }

} // namespace rocgraph
