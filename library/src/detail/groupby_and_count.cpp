// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "detail/graph_partition_utils.cuh"

#include "detail/shuffle_wrappers.hpp"
#include "graph_functions.hpp"
#include "partition_manager.hpp"
#include "utilities/host_scalar_comm.hpp"
#include "utilities/shuffle_comm_device.hpp"

#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scatter.h>
#include <thrust/tuple.h>

#include <tuple>

namespace rocgraph
{
    namespace detail
    {

        template <typename vertex_t, typename edge_t, typename weight_t, typename edge_type_t>
        rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
            raft::handle_t const&                            handle,
            rmm::device_uvector<vertex_t>&                   d_edgelist_majors,
            rmm::device_uvector<vertex_t>&                   d_edgelist_minors,
            std::optional<rmm::device_uvector<weight_t>>&    d_edgelist_weights,
            std::optional<rmm::device_uvector<edge_t>>&      d_edgelist_edge_ids,
            std::optional<rmm::device_uvector<edge_type_t>>& d_edgelist_edge_types,
            bool groupby_and_count_local_partition_by_minor)
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

            auto total_global_mem = handle.get_device_properties().totalGlobalMem;
            auto element_size     = sizeof(vertex_t) * 2
                                + (d_edgelist_weights ? sizeof(weight_t) : size_t{0})
                                + (d_edgelist_edge_ids ? sizeof(edge_t) : size_t{0})
                                + (d_edgelist_edge_types ? sizeof(edge_type_t) : size_t{0});
            auto constexpr mem_frugal_ratio
                = 0.1; // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
            // total_global_mem, switch to the memory frugal approach (thrust::sort is used to
            // group-by by default, and thrust::sort requires temporary buffer comparable to the input
            // data size)
            auto mem_frugal_threshold = static_cast<size_t>(
                static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);

            auto pair_first = thrust::make_zip_iterator(
                thrust::make_tuple(d_edgelist_majors.begin(), d_edgelist_minors.begin()));

            rmm::device_uvector<size_t> result(0, handle.get_stream());

            if(groupby_and_count_local_partition_by_minor)
            {
                auto local_edge_and_vertex_partition_id_pair_op =
                    [major_comm_size,
                     local_edge_partition_id_key_func
                     = rocgraph::detail::compute_local_edge_partition_id_from_ext_edge_endpoints_t<
                         vertex_t>{comm_size, major_comm_size, minor_comm_size},
                     vertex_partition_id_key_func
                     = rocgraph::detail::compute_vertex_partition_id_from_ext_vertex_t<vertex_t>{
                         comm_size}] __device__(auto pair) {
                        auto local_edge_partition_id = local_edge_partition_id_key_func(pair);
                        auto vertex_partition_id
                            = vertex_partition_id_key_func(thrust::get<1>(pair));
                        return (local_edge_partition_id * major_comm_size)
                               + ((vertex_partition_id) % major_comm_size);
                    };

                if(d_edgelist_weights)
                {
                    if(d_edgelist_edge_ids)
                    {
                        if(d_edgelist_edge_types)
                        {
                            result = rocgraph::groupby_and_count(
                                pair_first,
                                pair_first + d_edgelist_majors.size(),
                                thrust::make_zip_iterator(d_edgelist_weights->begin(),
                                                          d_edgelist_edge_ids->begin(),
                                                          d_edgelist_edge_types->begin()),
                                local_edge_and_vertex_partition_id_pair_op,
                                comm_size,
                                mem_frugal_threshold,
                                handle.get_stream());
                        }
                        else
                        {
                            result = rocgraph::groupby_and_count(
                                pair_first,
                                pair_first + d_edgelist_majors.size(),
                                thrust::make_zip_iterator(d_edgelist_weights->begin(),
                                                          d_edgelist_edge_ids->begin()),
                                local_edge_and_vertex_partition_id_pair_op,
                                comm_size,
                                mem_frugal_threshold,
                                handle.get_stream());
                        }
                    }
                    else
                    {
                        if(d_edgelist_edge_types)
                        {
                            result = rocgraph::groupby_and_count(
                                pair_first,
                                pair_first + d_edgelist_majors.size(),
                                thrust::make_zip_iterator(d_edgelist_weights->begin(),
                                                          d_edgelist_edge_types->begin()),
                                local_edge_and_vertex_partition_id_pair_op,
                                comm_size,
                                mem_frugal_threshold,
                                handle.get_stream());
                        }
                        else
                        {
                            result = rocgraph::groupby_and_count(
                                pair_first,
                                pair_first + d_edgelist_majors.size(),
                                d_edgelist_weights->begin(),
                                local_edge_and_vertex_partition_id_pair_op,
                                comm_size,
                                mem_frugal_threshold,
                                handle.get_stream());
                        }
                    }
                }
                else
                {
                    if(d_edgelist_edge_ids)
                    {
                        if(d_edgelist_edge_types)
                        {
                            result = rocgraph::groupby_and_count(
                                pair_first,
                                pair_first + d_edgelist_majors.size(),
                                thrust::make_zip_iterator(d_edgelist_edge_ids->begin(),
                                                          d_edgelist_edge_types->begin()),
                                local_edge_and_vertex_partition_id_pair_op,
                                comm_size,
                                mem_frugal_threshold,
                                handle.get_stream());
                        }
                        else
                        {
                            result = rocgraph::groupby_and_count(
                                pair_first,
                                pair_first + d_edgelist_majors.size(),
                                d_edgelist_edge_ids->begin(),
                                local_edge_and_vertex_partition_id_pair_op,
                                comm_size,
                                mem_frugal_threshold,
                                handle.get_stream());
                        }
                    }
                    else
                    {
                        if(d_edgelist_edge_types)
                        {
                            result = rocgraph::groupby_and_count(
                                pair_first,
                                pair_first + d_edgelist_majors.size(),
                                d_edgelist_edge_types->begin(),
                                local_edge_and_vertex_partition_id_pair_op,
                                comm_size,
                                mem_frugal_threshold,
                                handle.get_stream());
                        }
                        else
                        {
                            result = rocgraph::groupby_and_count(
                                pair_first,
                                pair_first + d_edgelist_majors.size(),
                                local_edge_and_vertex_partition_id_pair_op,
                                comm_size,
                                mem_frugal_threshold,
                                handle.get_stream());
                        }
                    }
                }
            }
            else
            {
                auto local_edge_partition_id_op =
                    [key_func
                     = rocgraph::detail::compute_local_edge_partition_id_from_ext_edge_endpoints_t<
                         vertex_t>{
                         comm_size, major_comm_size, minor_comm_size}] __device__(auto pair) {
                        return key_func(pair);
                    };

                if(d_edgelist_weights)
                {
                    if(d_edgelist_edge_ids)
                    {
                        if(d_edgelist_edge_types)
                        {
                            result = rocgraph::groupby_and_count(
                                pair_first,
                                pair_first + d_edgelist_majors.size(),
                                thrust::make_zip_iterator(d_edgelist_weights->begin(),
                                                          d_edgelist_edge_ids->begin(),
                                                          d_edgelist_edge_types->begin()),
                                local_edge_partition_id_op,
                                comm_size,
                                mem_frugal_threshold,
                                handle.get_stream());
                        }
                        else
                        {
                            result = rocgraph::groupby_and_count(
                                pair_first,
                                pair_first + d_edgelist_majors.size(),
                                thrust::make_zip_iterator(d_edgelist_weights->begin(),
                                                          d_edgelist_edge_ids->begin()),
                                local_edge_partition_id_op,
                                comm_size,
                                mem_frugal_threshold,
                                handle.get_stream());
                        }
                    }
                    else
                    {
                        if(d_edgelist_edge_types)
                        {
                            result = rocgraph::groupby_and_count(
                                pair_first,
                                pair_first + d_edgelist_majors.size(),
                                thrust::make_zip_iterator(d_edgelist_weights->begin(),
                                                          d_edgelist_edge_types->begin()),
                                local_edge_partition_id_op,
                                comm_size,
                                mem_frugal_threshold,
                                handle.get_stream());
                        }
                        else
                        {
                            result
                                = rocgraph::groupby_and_count(pair_first,
                                                              pair_first + d_edgelist_majors.size(),
                                                              d_edgelist_weights->begin(),
                                                              local_edge_partition_id_op,
                                                              comm_size,
                                                              mem_frugal_threshold,
                                                              handle.get_stream());
                        }
                    }
                }
                else
                {
                    if(d_edgelist_edge_ids)
                    {
                        if(d_edgelist_edge_types)
                        {
                            result = rocgraph::groupby_and_count(
                                pair_first,
                                pair_first + d_edgelist_majors.size(),
                                thrust::make_zip_iterator(d_edgelist_edge_ids->begin(),
                                                          d_edgelist_edge_types->begin()),
                                local_edge_partition_id_op,
                                comm_size,
                                mem_frugal_threshold,
                                handle.get_stream());
                        }
                        else
                        {
                            result
                                = rocgraph::groupby_and_count(pair_first,
                                                              pair_first + d_edgelist_majors.size(),
                                                              d_edgelist_edge_ids->begin(),
                                                              local_edge_partition_id_op,
                                                              comm_size,
                                                              mem_frugal_threshold,
                                                              handle.get_stream());
                        }
                    }
                    else
                    {
                        if(d_edgelist_edge_types)
                        {
                            result
                                = rocgraph::groupby_and_count(pair_first,
                                                              pair_first + d_edgelist_majors.size(),
                                                              d_edgelist_edge_types->begin(),
                                                              local_edge_partition_id_op,
                                                              comm_size,
                                                              mem_frugal_threshold,
                                                              handle.get_stream());
                        }
                        else
                        {
                            result
                                = rocgraph::groupby_and_count(pair_first,
                                                              pair_first + d_edgelist_majors.size(),
                                                              local_edge_partition_id_op,
                                                              comm_size,
                                                              mem_frugal_threshold,
                                                              handle.get_stream());
                        }
                    }
                }
            }

            return result;
        }

        template rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
            raft::handle_t const&                        handle,
            rmm::device_uvector<int32_t>&                d_edgelist_majors,
            rmm::device_uvector<int32_t>&                d_edgelist_minors,
            std::optional<rmm::device_uvector<float>>&   d_edgelist_weights,
            std::optional<rmm::device_uvector<int32_t>>& d_edgelist_edge_ids,
            std::optional<rmm::device_uvector<int32_t>>& d_edgelist_edge_types,
            bool                                         groupby_and_counts_local_partition);

        template rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
            raft::handle_t const&                        handle,
            rmm::device_uvector<int32_t>&                d_edgelist_majors,
            rmm::device_uvector<int32_t>&                d_edgelist_minors,
            std::optional<rmm::device_uvector<double>>&  d_edgelist_weights,
            std::optional<rmm::device_uvector<int32_t>>& d_edgelist_edge_ids,
            std::optional<rmm::device_uvector<int32_t>>& d_edgelist_edge_types,
            bool                                         groupby_and_counts_local_partition);

        template rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
            raft::handle_t const&                        handle,
            rmm::device_uvector<int32_t>&                d_edgelist_majors,
            rmm::device_uvector<int32_t>&                d_edgelist_minors,
            std::optional<rmm::device_uvector<float>>&   d_edgelist_weights,
            std::optional<rmm::device_uvector<int64_t>>& d_edgelist_edge_ids,
            std::optional<rmm::device_uvector<int32_t>>& d_edgelist_edge_types,
            bool                                         groupby_and_counts_local_partition);

        template rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
            raft::handle_t const&                        handle,
            rmm::device_uvector<int32_t>&                d_edgelist_majors,
            rmm::device_uvector<int32_t>&                d_edgelist_minors,
            std::optional<rmm::device_uvector<double>>&  d_edgelist_weights,
            std::optional<rmm::device_uvector<int64_t>>& d_edgelist_edge_ids,
            std::optional<rmm::device_uvector<int32_t>>& d_edgelist_edge_types,
            bool                                         groupby_and_counts_local_partition);

        template rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
            raft::handle_t const&                        handle,
            rmm::device_uvector<int64_t>&                d_edgelist_majors,
            rmm::device_uvector<int64_t>&                d_edgelist_minors,
            std::optional<rmm::device_uvector<float>>&   d_edgelist_weights,
            std::optional<rmm::device_uvector<int64_t>>& d_edgelist_edge_ids,
            std::optional<rmm::device_uvector<int32_t>>& d_edgelist_edge_types,
            bool                                         groupby_and_counts_local_partition);

        template rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
            raft::handle_t const&                        handle,
            rmm::device_uvector<int64_t>&                d_edgelist_majors,
            rmm::device_uvector<int64_t>&                d_edgelist_minors,
            std::optional<rmm::device_uvector<double>>&  d_edgelist_weights,
            std::optional<rmm::device_uvector<int64_t>>& d_edgelist_edge_ids,
            std::optional<rmm::device_uvector<int32_t>>& d_edgelist_edge_types,
            bool                                         groupby_and_counts_local_partition);

    } // namespace detail
} // namespace rocgraph
