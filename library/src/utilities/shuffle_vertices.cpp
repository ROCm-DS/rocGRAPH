// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "detail/graph_partition_utils.cuh"

#include "detail/shuffle_wrappers.hpp"
#include "utilities/shuffle_comm_device.hpp"

#include <thrust/tuple.h>

#include <tuple>

namespace rocgraph
{

    namespace
    {

        template <typename vertex_t, typename func_t>
        rmm::device_uvector<vertex_t> shuffle_vertices_by_gpu_id_impl(
            raft::handle_t const& handle, rmm::device_uvector<vertex_t>&& d_vertices, func_t func)
        {
            rmm::device_uvector<vertex_t> d_rx_vertices(0, handle.get_stream());
            std::tie(d_rx_vertices, std::ignore) = rocgraph::groupby_gpu_id_and_shuffle_values(
                handle.get_comms(),
                d_vertices.begin(),
                d_vertices.end(),
                [key_func = func] __device__(auto val) { return key_func(val); },
                handle.get_stream());

            return d_rx_vertices;
        }

        template <typename vertex_t, typename value_t, typename func_t>
        std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<value_t>>
            shuffle_vertices_and_values_by_gpu_id_impl(raft::handle_t const&           handle,
                                                       rmm::device_uvector<vertex_t>&& d_vertices,
                                                       rmm::device_uvector<value_t>&&  d_values,
                                                       func_t                          func)
        {
            std::tie(d_vertices, d_values, std::ignore)
                = rocgraph::groupby_gpu_id_and_shuffle_kv_pairs(
                    handle.get_comms(),
                    d_vertices.begin(),
                    d_vertices.end(),
                    d_values.begin(),
                    [key_func = func] __device__(auto val) { return key_func(val); },
                    handle.get_stream());

            return std::make_tuple(std::move(d_vertices), std::move(d_values));
        }

    } // namespace

    namespace detail
    {

        template <typename vertex_t>
        rmm::device_uvector<vertex_t> shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
            raft::handle_t const& handle, rmm::device_uvector<vertex_t>&& vertices)
        {
            auto const comm_size = handle.get_comms().get_size();
            auto& major_comm = handle.get_subcomm(rocgraph::partition_manager::major_comm_name());
            auto const major_comm_size = major_comm.get_size();
            auto& minor_comm = handle.get_subcomm(rocgraph::partition_manager::minor_comm_name());
            auto const minor_comm_size = minor_comm.get_size();

            return shuffle_vertices_by_gpu_id_impl(
                handle,
                std::move(vertices),
                rocgraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{
                    comm_size, major_comm_size, minor_comm_size});
        }

        template <typename vertex_t, typename value_t>
        std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<value_t>>
            shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
                raft::handle_t const&           handle,
                rmm::device_uvector<vertex_t>&& vertices,
                rmm::device_uvector<value_t>&&  values)
        {
            auto const comm_size = handle.get_comms().get_size();
            auto& major_comm = handle.get_subcomm(rocgraph::partition_manager::major_comm_name());
            auto const major_comm_size = major_comm.get_size();
            auto& minor_comm = handle.get_subcomm(rocgraph::partition_manager::minor_comm_name());
            auto const minor_comm_size = minor_comm.get_size();

            return shuffle_vertices_and_values_by_gpu_id_impl(
                handle,
                std::move(vertices),
                std::move(values),
                rocgraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{
                    comm_size, major_comm_size, minor_comm_size});
        }

        template <typename vertex_t>
        rmm::device_uvector<vertex_t> shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
            raft::handle_t const&           handle,
            rmm::device_uvector<vertex_t>&& vertices,
            std::vector<vertex_t> const&    vertex_partition_range_lasts)
        {
            rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(
                vertex_partition_range_lasts.size(), handle.get_stream());
            raft::update_device(d_vertex_partition_range_lasts.data(),
                                vertex_partition_range_lasts.data(),
                                vertex_partition_range_lasts.size(),
                                handle.get_stream());
            auto& major_comm = handle.get_subcomm(rocgraph::partition_manager::major_comm_name());
            auto const major_comm_size = major_comm.get_size();
            auto& minor_comm = handle.get_subcomm(rocgraph::partition_manager::minor_comm_name());
            auto const minor_comm_size = minor_comm.get_size();

            auto return_value = shuffle_vertices_by_gpu_id_impl(
                handle,
                std::move(vertices),
                rocgraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
                    raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                                      d_vertex_partition_range_lasts.size()),
                    major_comm_size,
                    minor_comm_size});

            handle.sync_stream();

            return return_value;
        }

        template <typename vertex_t, typename value_t>
        std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<value_t>>
            shuffle_int_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
                raft::handle_t const&           handle,
                rmm::device_uvector<vertex_t>&& vertices,
                rmm::device_uvector<value_t>&&  values,
                std::vector<vertex_t> const&    vertex_partition_range_lasts)
        {
            rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(
                vertex_partition_range_lasts.size(), handle.get_stream());
            raft::update_device(d_vertex_partition_range_lasts.data(),
                                vertex_partition_range_lasts.data(),
                                vertex_partition_range_lasts.size(),
                                handle.get_stream());
            auto& major_comm = handle.get_subcomm(rocgraph::partition_manager::major_comm_name());
            auto const major_comm_size = major_comm.get_size();
            auto& minor_comm = handle.get_subcomm(rocgraph::partition_manager::minor_comm_name());
            auto const minor_comm_size = minor_comm.get_size();

            auto return_value = shuffle_vertices_and_values_by_gpu_id_impl(
                handle,
                std::move(vertices),
                std::move(values),
                rocgraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
                    raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                                      d_vertex_partition_range_lasts.size()),
                    major_comm_size,
                    minor_comm_size});

            return return_value;
        }

        template rmm::device_uvector<int32_t>
            shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
                raft::handle_t const&          handle,
                rmm::device_uvector<int32_t>&& vertices,
                std::vector<int32_t> const&    vertex_partition_range_lasts);

        template rmm::device_uvector<int64_t>
            shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
                raft::handle_t const&          handle,
                rmm::device_uvector<int64_t>&& vertices,
                std::vector<int64_t> const&    vertex_partition_range_lasts);

        template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
            shuffle_int_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
                raft::handle_t const&          handle,
                rmm::device_uvector<int32_t>&& d_vertices,
                rmm::device_uvector<int32_t>&& d_values,
                std::vector<int32_t> const&    vertex_partition_range_lasts);

        template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>
            shuffle_int_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
                raft::handle_t const&          handle,
                rmm::device_uvector<int64_t>&& d_vertices,
                rmm::device_uvector<int32_t>&& d_values,
                std::vector<int64_t> const&    vertex_partition_range_lasts);

        template rmm::device_uvector<int32_t>
            shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
                raft::handle_t const& handle, rmm::device_uvector<int32_t>&& d_vertices);

        template rmm::device_uvector<int64_t>
            shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
                raft::handle_t const& handle, rmm::device_uvector<int64_t>&& d_vertices);

        template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
            shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
                raft::handle_t const&          handle,
                rmm::device_uvector<int32_t>&& vertices,
                rmm::device_uvector<int32_t>&& values);

        template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<size_t>>
            shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
                raft::handle_t const&          handle,
                rmm::device_uvector<int32_t>&& vertices,
                rmm::device_uvector<size_t>&&  values);

        template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<float>>
            shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
                raft::handle_t const&          handle,
                rmm::device_uvector<int32_t>&& vertices,
                rmm::device_uvector<float>&&   values);

        template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<double>>
            shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
                raft::handle_t const&          handle,
                rmm::device_uvector<int32_t>&& vertices,
                rmm::device_uvector<double>&&  values);

        template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>
            shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
                raft::handle_t const&          handle,
                rmm::device_uvector<int64_t>&& vertices,
                rmm::device_uvector<int32_t>&& values);

        template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
            shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
                raft::handle_t const&          handle,
                rmm::device_uvector<int64_t>&& vertices,
                rmm::device_uvector<int64_t>&& values);

        template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<size_t>>
            shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
                raft::handle_t const&          handle,
                rmm::device_uvector<int64_t>&& vertices,
                rmm::device_uvector<size_t>&&  values);

        template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<float>>
            shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
                raft::handle_t const&          handle,
                rmm::device_uvector<int64_t>&& vertices,
                rmm::device_uvector<float>&&   values);

        template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<double>>
            shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
                raft::handle_t const&          handle,
                rmm::device_uvector<int64_t>&& vertices,
                rmm::device_uvector<double>&&  values);

    } // namespace detail

    template <typename vertex_t, typename value_t>
    std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<value_t>>
        shuffle_external_vertex_value_pairs(raft::handle_t const&           handle,
                                            rmm::device_uvector<vertex_t>&& vertices,
                                            rmm::device_uvector<value_t>&&  values)
    {
        return detail::shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
            handle, std::move(vertices), std::move(values));
    }

    template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
        shuffle_external_vertex_value_pairs(raft::handle_t const&          handle,
                                            rmm::device_uvector<int32_t>&& vertices,
                                            rmm::device_uvector<int32_t>&& values);

    template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<size_t>>
        shuffle_external_vertex_value_pairs(raft::handle_t const&          handle,
                                            rmm::device_uvector<int32_t>&& vertices,
                                            rmm::device_uvector<size_t>&&  values);

    template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<float>>
        shuffle_external_vertex_value_pairs(raft::handle_t const&          handle,
                                            rmm::device_uvector<int32_t>&& vertices,
                                            rmm::device_uvector<float>&&   values);

    template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<double>>
        shuffle_external_vertex_value_pairs(raft::handle_t const&          handle,
                                            rmm::device_uvector<int32_t>&& vertices,
                                            rmm::device_uvector<double>&&  values);

    template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>
        shuffle_external_vertex_value_pairs(raft::handle_t const&          handle,
                                            rmm::device_uvector<int64_t>&& vertices,
                                            rmm::device_uvector<int32_t>&& values);

    template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
        shuffle_external_vertex_value_pairs(raft::handle_t const&          handle,
                                            rmm::device_uvector<int64_t>&& vertices,
                                            rmm::device_uvector<int64_t>&& values);

    template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<size_t>>
        shuffle_external_vertex_value_pairs(raft::handle_t const&          handle,
                                            rmm::device_uvector<int64_t>&& vertices,
                                            rmm::device_uvector<size_t>&&  values);

    template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<float>>
        shuffle_external_vertex_value_pairs(raft::handle_t const&          handle,
                                            rmm::device_uvector<int64_t>&& vertices,
                                            rmm::device_uvector<float>&&   values);

    template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<double>>
        shuffle_external_vertex_value_pairs(raft::handle_t const&          handle,
                                            rmm::device_uvector<int64_t>&& vertices,
                                            rmm::device_uvector<double>&&  values);

    template <typename vertex_t>
    rmm::device_uvector<vertex_t>
        shuffle_external_vertices(raft::handle_t const&           handle,
                                  rmm::device_uvector<vertex_t>&& vertices)
    {
        return detail::shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
            handle, std::move(vertices));
    }

    template rmm::device_uvector<int32_t>
        shuffle_external_vertices(raft::handle_t const&          handle,
                                  rmm::device_uvector<int32_t>&& d_vertices);

    template rmm::device_uvector<int64_t>
        shuffle_external_vertices(raft::handle_t const&          handle,
                                  rmm::device_uvector<int64_t>&& d_vertices);

} // namespace rocgraph
