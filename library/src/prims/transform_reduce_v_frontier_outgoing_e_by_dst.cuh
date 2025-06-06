// Copyright (C) 2020-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "detail/graph_partition_utils.cuh"
#include "prims/detail/extract_transform_v_frontier_e.cuh"
#include "prims/property_op_utils.cuh"
#include "prims/reduce_op.cuh"

#include "edge_partition_device_view_device.hpp"
#include "edge_partition_endpoint_property_device_view_device.hpp"
#include "edge_src_dst_property.hpp"
#include "graph_view.hpp"
#include "partition_manager.hpp"
#include "utilities/dataframe_buffer.hpp"
#include "utilities/device_comm.hpp"
#include "utilities/device_functors_device.hpp"
#include "utilities/error.hpp"
#include "utilities/host_scalar_comm.hpp"
#include "utilities/shuffle_comm_device.hpp"
#include "vertex_partition_device_view_device.hpp"

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/optional.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/type_traits/integer_sequence.h>
#include <thrust/unique.h>

#include <hipcub/hipcub.hpp>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace rocgraph
{

    namespace detail
    {

        int32_t constexpr update_v_frontier_from_outgoing_e_kernel_block_size = 512;

        template <typename key_t,
                  typename payload_t,
                  typename vertex_t,
                  typename src_value_t,
                  typename dst_value_t,
                  typename e_value_t,
                  typename EdgeOp>
        struct transform_reduce_v_frontier_call_e_op_t
        {
            EdgeOp e_op{};

            __host__ __device__ thrust::optional<std::conditional_t<
                !std::is_same_v<key_t, void> && !std::is_same_v<payload_t, void>,
                thrust::tuple<key_t, payload_t>,
                std::conditional_t<!std::is_same_v<key_t, void>, key_t, payload_t>>>
                                operator()(
                    key_t key, vertex_t dst, src_value_t sv, dst_value_t dv, e_value_t ev) const
            {
                auto e_op_result = e_op(key, dst, sv, dv, ev);
                if(e_op_result.has_value())
                {
                    if constexpr(std::is_same_v<key_t, vertex_t> && std::is_same_v<payload_t, void>)
                    {
                        return dst;
                    }
                    else if constexpr(std::is_same_v<key_t, vertex_t>
                                      && !std::is_same_v<payload_t, void>)
                    {
                        return thrust::make_tuple(dst, *e_op_result);
                    }
                    else if constexpr(!std::is_same_v<key_t, vertex_t>
                                      && std::is_same_v<payload_t, void>)
                    {
                        return thrust::make_tuple(dst, *e_op_result);
                    }
                    else
                    {
                        return thrust::make_tuple(
                            thrust::make_tuple(dst, thrust::get<0>(*e_op_result)),
                            thrust::get<1>(*e_op_result));
                    }
                }
                else
                {
                    return thrust::nullopt;
                }
            }
        };

        template <typename key_t, typename payload_t, typename ReduceOp>
        auto sort_and_reduce_buffer_elements(
            raft::handle_t const&                                                    handle,
            decltype(allocate_dataframe_buffer<key_t>(0, rmm::cuda_stream_view{}))&& key_buffer,
            decltype(allocate_optional_dataframe_buffer<payload_t>(
                0, rmm::cuda_stream_view{}))&&                                       payload_buffer,
            ReduceOp                                                                 reduce_op)
        {
            if constexpr(std::is_same_v<payload_t, void>)
            {
                thrust::sort(handle.get_thrust_policy(),
                             get_dataframe_buffer_begin(key_buffer),
                             get_dataframe_buffer_end(key_buffer));
            }
            else
            {
                thrust::sort_by_key(handle.get_thrust_policy(),
                                    get_dataframe_buffer_begin(key_buffer),
                                    get_dataframe_buffer_end(key_buffer),
                                    get_optional_dataframe_buffer_begin<payload_t>(payload_buffer));
            }

            if constexpr(std::is_same_v<payload_t, void>)
            {
                auto it = thrust::unique(handle.get_thrust_policy(),
                                         get_dataframe_buffer_begin(key_buffer),
                                         get_dataframe_buffer_end(key_buffer));
                resize_dataframe_buffer(key_buffer,
                                        static_cast<size_t>(thrust::distance(
                                            get_dataframe_buffer_begin(key_buffer), it)),
                                        handle.get_stream());
                shrink_to_fit_dataframe_buffer(key_buffer, handle.get_stream());
            }
            else if constexpr(std::is_same_v<ReduceOp,
                                             reduce_op::any<typename ReduceOp::value_type>>)
            {
                auto it = thrust::unique_by_key(
                    handle.get_thrust_policy(),
                    get_dataframe_buffer_begin(key_buffer),
                    get_dataframe_buffer_end(key_buffer),
                    get_optional_dataframe_buffer_begin<payload_t>(payload_buffer));
                resize_dataframe_buffer(
                    key_buffer,
                    static_cast<size_t>(thrust::distance(get_dataframe_buffer_begin(key_buffer),
                                                         thrust::get<0>(it))),
                    handle.get_stream());
                resize_dataframe_buffer(
                    payload_buffer, size_dataframe_buffer(key_buffer), handle.get_stream());
                shrink_to_fit_dataframe_buffer(key_buffer, handle.get_stream());
                shrink_to_fit_dataframe_buffer(payload_buffer, handle.get_stream());
            }
            else
            {
                auto num_uniques = thrust::count_if(
                    handle.get_thrust_policy(),
                    thrust::make_counting_iterator(size_t{0}),
                    thrust::make_counting_iterator(size_dataframe_buffer(key_buffer)),
                    is_first_in_run_t<decltype(get_dataframe_buffer_begin(key_buffer))>{
                        get_dataframe_buffer_begin(key_buffer)});

                auto new_key_buffer
                    = allocate_dataframe_buffer<key_t>(num_uniques, handle.get_stream());
                auto new_payload_buffer
                    = allocate_dataframe_buffer<payload_t>(num_uniques, handle.get_stream());

                thrust::reduce_by_key(
                    handle.get_thrust_policy(),
                    get_dataframe_buffer_begin(key_buffer),
                    get_dataframe_buffer_end(key_buffer),
                    get_optional_dataframe_buffer_begin<payload_t>(payload_buffer),
                    get_dataframe_buffer_begin(new_key_buffer),
                    get_dataframe_buffer_begin(new_payload_buffer),
                    thrust::equal_to<key_t>(),
                    reduce_op);

                key_buffer     = std::move(new_key_buffer);
                payload_buffer = std::move(new_payload_buffer);
            }

            return std::make_tuple(std::move(key_buffer), std::move(payload_buffer));
        }

    } // namespace detail

    template <typename GraphViewType, typename VertexFrontierBucketType>
    size_t compute_num_out_nbrs_from_frontier(raft::handle_t const&           handle,
                                              GraphViewType const&            graph_view,
                                              VertexFrontierBucketType const& frontier)
    {
        static_assert(!GraphViewType::is_storage_transposed,
                      "GraphViewType should support the push model.");

        using vertex_t = typename GraphViewType::vertex_type;
        using edge_t   = typename GraphViewType::edge_type;
        using key_t    = typename VertexFrontierBucketType::key_type;

        size_t ret{0};

        vertex_t const* local_frontier_vertex_first{nullptr};
        if constexpr(std::is_same_v<key_t, vertex_t>)
        {
            local_frontier_vertex_first = frontier.begin();
        }
        else
        {
            local_frontier_vertex_first = thrust::get<0>(frontier.begin().get_iterator_tuple());
        }

        std::vector<size_t> local_frontier_sizes{};
        if constexpr(GraphViewType::is_multi_gpu)
        {
            auto& minor_comm = handle.get_subcomm(rocgraph::partition_manager::minor_comm_name());
            local_frontier_sizes
                = host_scalar_allgather(minor_comm, frontier.size(), handle.get_stream());
        }
        else
        {
            local_frontier_sizes = std::vector<size_t>{static_cast<size_t>(frontier.size())};
        }

        auto edge_mask_view = graph_view.edge_mask_view();

        for(size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i)
        {
            auto edge_partition
                = edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
                    graph_view.local_edge_partition_view(i));
            auto edge_partition_e_mask
                = edge_mask_view
                      ? thrust::make_optional<detail::edge_partition_edge_property_device_view_t<
                            edge_t,
                            packed_bool_container_t const*,
                            bool>>(*edge_mask_view, i)
                      : thrust::nullopt;

            if constexpr(GraphViewType::is_multi_gpu)
            {
                auto& minor_comm
                    = handle.get_subcomm(rocgraph::partition_manager::minor_comm_name());
                auto const minor_comm_rank = minor_comm.get_rank();

                rmm::device_uvector<vertex_t> edge_partition_frontier_vertices(
                    local_frontier_sizes[i], handle.get_stream());
                device_bcast(minor_comm,
                             local_frontier_vertex_first,
                             edge_partition_frontier_vertices.data(),
                             local_frontier_sizes[i],
                             static_cast<int>(i),
                             handle.get_stream());

                if(edge_partition_e_mask)
                {
                    ret += edge_partition.compute_number_of_edges_with_mask(
                        (*edge_partition_e_mask).value_first(),
                        edge_partition_frontier_vertices.begin(),
                        edge_partition_frontier_vertices.end(),
                        handle.get_stream());
                }
                else
                {
                    ret += edge_partition.compute_number_of_edges(
                        edge_partition_frontier_vertices.begin(),
                        edge_partition_frontier_vertices.end(),
                        handle.get_stream());
                }
            }
            else
            {
                assert(i == 0);
                if(edge_partition_e_mask)
                {
                    ret += edge_partition.compute_number_of_edges_with_mask(
                        (*edge_partition_e_mask).value_first(),
                        local_frontier_vertex_first,
                        local_frontier_vertex_first + frontier.size(),
                        handle.get_stream());
                }
                else
                {
                    ret += edge_partition.compute_number_of_edges(local_frontier_vertex_first,
                                                                  local_frontier_vertex_first
                                                                      + frontier.size(),
                                                                  handle.get_stream());
                }
            }
        }

        return ret;
    }

    /**
 * @brief Iterate over outgoing edges from the current vertex frontier and reduce valid edge functor
 * outputs by (tagged-)destination ID.
 *
 * Edge functor outputs are thrust::optional objects and invalid if thrust::nullopt. Vertices are
 * assumed to be tagged if VertexFrontierBucketType::key_type is a tuple of a vertex type and a tag
 * type (VertexFrontierBucketType::key_type is identical to a vertex type otherwise).
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexFrontierBucketType Type of the vertex frontier bucket class which abstracts the
 * current (tagged-)vertex frontier.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @tparam ReduceOp Type of the binary reduction operator.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param frontier VertexFrontierBucketType class object for the current vertex frontier.
 * @param edge_src_value_input Wrapper used to access source input property values (for the edge
 * sources assigned to this process in multi-GPU). Use either rocgraph::edge_src_property_t::view()
 * (if @p e_op needs to access source property values) or rocgraph::edge_src_dummy_property_t::view()
 * (if @p e_op does not access source property values). Use update_edge_src_property to fill the
 * wrapper.
 * @param edge_dst_value_input Wrapper used to access destination input property values (for the
 * edge destinations assigned to this process in multi-GPU). Use either
 * rocgraph::edge_dst_property_t::view() (if @p e_op needs to access destination property values) or
 * rocgraph::edge_dst_dummy_property_t::view() (if @p e_op does not access destination property
 * values). Use update_edge_dst_property to fill the wrapper.
 * @param edge_value_input Wrapper used to access edge input property values (for the edges assigned
 * to this process in multi-GPU). Use either rocgraph::edge_property_t::view() (if @p e_op needs to
 * access edge property values) or rocgraph::edge_dummy_property_t::view() (if @p e_op does not
 * access edge property values).
 * @param e_op Quinary operator takes edge source, edge destination, property values for the source,
 * destination, and edge and returns 1) thrust::nullopt (if invalid and to be discarded); 2) dummy
 * (but valid) thrust::optional object (e.g. thrust::optional<std::byte>{std::byte{0}}, if vertices
 * are not tagged and ReduceOp::value_type is void); 3) a tag (if vertices are tagged and
 * ReduceOp::value_type is void); 4) a value to be reduced using the @p reduce_op (if vertices are
 * not tagged and ReduceOp::value_type is not void); or 5) a tuple of a tag and a value to be
 * reduced (if vertices are tagged and ReduceOp::value_type is not void).
 * @param reduce_op Binary operator that takes two input arguments and reduce the two values to one.
 * There are pre-defined reduction operators in prims/reduce_op.cuh. It is
 * recommended to use the pre-defined reduction operators whenever possible as the current (and
 * future) implementations of graph primitives may check whether @p ReduceOp is a known type (or has
 * known member variables) to take a more optimized code path. See the documentation in the
 * reduce_op.cuh file for instructions on writing custom reduction operators.
 * @return Tuple of key values and payload values (if ReduceOp::value_type is not void) or just key
 * values (if ReduceOp::value_type is void). Keys in the return values are sorted in ascending order
 * using a vertex ID as the primary key and a tag (if relevant) as the secondary key.
 */
    template <typename GraphViewType,
              typename VertexFrontierBucketType,
              typename EdgeSrcValueInputWrapper,
              typename EdgeDstValueInputWrapper,
              typename EdgeValueInputWrapper,
              typename EdgeOp,
              typename ReduceOp>
    std::conditional_t<
        !std::is_same_v<typename ReduceOp::value_type, void>,
        std::tuple<decltype(allocate_dataframe_buffer<typename VertexFrontierBucketType::key_type>(
                       0, rmm::cuda_stream_view{})),
                   decltype(detail::allocate_optional_dataframe_buffer<
                            typename ReduceOp::value_type>(0, rmm::cuda_stream_view{}))>,
        decltype(allocate_dataframe_buffer<typename VertexFrontierBucketType::key_type>(
            0, rmm::cuda_stream_view{}))>
        transform_reduce_v_frontier_outgoing_e_by_dst(raft::handle_t const&           handle,
                                                      GraphViewType const&            graph_view,
                                                      VertexFrontierBucketType const& frontier,
                                                      EdgeSrcValueInputWrapper edge_src_value_input,
                                                      EdgeDstValueInputWrapper edge_dst_value_input,
                                                      EdgeValueInputWrapper    edge_value_input,
                                                      EdgeOp                   e_op,
                                                      ReduceOp                 reduce_op,
                                                      bool do_expensive_check = false)
    {
        static_assert(!GraphViewType::is_storage_transposed,
                      "GraphViewType should support the push model.");

        using vertex_t  = typename GraphViewType::vertex_type;
        using edge_t    = typename GraphViewType::edge_type;
        using key_t     = typename VertexFrontierBucketType::key_type;
        using payload_t = typename ReduceOp::value_type;

        if(do_expensive_check)
        {
            // currently, nothing to do
        }

        // 1. fill the buffer

        detail::transform_reduce_v_frontier_call_e_op_t<
            key_t,
            payload_t,
            vertex_t,
            typename EdgeSrcValueInputWrapper::value_type,
            typename EdgeDstValueInputWrapper::value_type,
            typename EdgeValueInputWrapper::value_type,
            EdgeOp>
            e_op_wrapper{e_op};

        auto [key_buffer, payload_buffer]
            = detail::extract_transform_v_frontier_e<false, key_t, payload_t>(handle,
                                                                              graph_view,
                                                                              frontier,
                                                                              edge_src_value_input,
                                                                              edge_dst_value_input,
                                                                              edge_value_input,
                                                                              e_op_wrapper,
                                                                              do_expensive_check);

        // 2. reduce the buffer

        std::tie(key_buffer, payload_buffer)
            = detail::sort_and_reduce_buffer_elements<key_t, payload_t, ReduceOp>(
                handle, std::move(key_buffer), std::move(payload_buffer), reduce_op);
        if constexpr(GraphViewType::is_multi_gpu)
        {
            // FIXME: this step is unnecessary if major_comm_size== 1
            auto& comm       = handle.get_comms();
            auto& major_comm = handle.get_subcomm(rocgraph::partition_manager::major_comm_name());
            auto const major_comm_rank = major_comm.get_rank();
            auto const major_comm_size = major_comm.get_size();
            auto& minor_comm = handle.get_subcomm(rocgraph::partition_manager::minor_comm_name());
            auto const minor_comm_rank = minor_comm.get_rank();
            auto const minor_comm_size = minor_comm.get_size();

            std::vector<vertex_t> h_vertex_lasts(major_comm_size);
            for(size_t i = 0; i < h_vertex_lasts.size(); ++i)
            {
                auto minor_range_vertex_partition_id
                    = detail::compute_local_edge_partition_minor_range_vertex_partition_id_t{
                        major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
                h_vertex_lasts[i]
                    = graph_view.vertex_partition_range_last(minor_range_vertex_partition_id);
            }

            rmm::device_uvector<vertex_t> d_vertex_lasts(h_vertex_lasts.size(),
                                                         handle.get_stream());
            raft::update_device(d_vertex_lasts.data(),
                                h_vertex_lasts.data(),
                                h_vertex_lasts.size(),
                                handle.get_stream());
            rmm::device_uvector<edge_t> d_tx_buffer_last_boundaries(d_vertex_lasts.size(),
                                                                    handle.get_stream());
            vertex_t const*             dst_first{nullptr};
            if constexpr(std::is_same_v<key_t, vertex_t>)
            {
                dst_first = get_dataframe_buffer_begin(key_buffer);
            }
            else
            {
                dst_first
                    = thrust::get<0>(get_dataframe_buffer_begin(key_buffer).get_iterator_tuple());
            }
            thrust::lower_bound(handle.get_thrust_policy(),
                                dst_first,
                                dst_first + size_dataframe_buffer(key_buffer),
                                d_vertex_lasts.begin(),
                                d_vertex_lasts.end(),
                                d_tx_buffer_last_boundaries.begin());
            std::vector<edge_t> h_tx_buffer_last_boundaries(d_tx_buffer_last_boundaries.size());
            raft::update_host(h_tx_buffer_last_boundaries.data(),
                              d_tx_buffer_last_boundaries.data(),
                              d_tx_buffer_last_boundaries.size(),
                              handle.get_stream());
            handle.sync_stream();
            std::vector<size_t> tx_counts(h_tx_buffer_last_boundaries.size());
            std::adjacent_difference(h_tx_buffer_last_boundaries.begin(),
                                     h_tx_buffer_last_boundaries.end(),
                                     tx_counts.begin());

            auto rx_key_buffer = allocate_dataframe_buffer<key_t>(size_t{0}, handle.get_stream());
            std::tie(rx_key_buffer, std::ignore) = shuffle_values(
                major_comm, get_dataframe_buffer_begin(key_buffer), tx_counts, handle.get_stream());
            key_buffer = std::move(rx_key_buffer);

            if constexpr(!std::is_same_v<payload_t, void>)
            {
                auto rx_payload_buffer
                    = allocate_dataframe_buffer<payload_t>(size_t{0}, handle.get_stream());
                std::tie(rx_payload_buffer, std::ignore)
                    = shuffle_values(major_comm,
                                     get_dataframe_buffer_begin(payload_buffer),
                                     tx_counts,
                                     handle.get_stream());
                payload_buffer = std::move(rx_payload_buffer);
            }

            std::tie(key_buffer, payload_buffer)
                = detail::sort_and_reduce_buffer_elements<key_t, payload_t, ReduceOp>(
                    handle, std::move(key_buffer), std::move(payload_buffer), reduce_op);
        }

        if constexpr(!std::is_same_v<payload_t, void>)
        {
            return std::make_tuple(std::move(key_buffer), std::move(payload_buffer));
        }
        else
        {
            return std::move(key_buffer);
        }
    }

} // namespace rocgraph
