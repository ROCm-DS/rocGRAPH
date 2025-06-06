// Copyright (C) 2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "prims/fill_edge_property.cuh"
#include "prims/transform_e.cuh"
#include "prims/update_edge_src_dst_property.cuh"

#include "algorithms.hpp"

#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

namespace rocgraph
{

    namespace detail
    {

        template <typename vertex_t, typename edge_t, bool multi_gpu>
        rmm::device_uvector<vertex_t> vertex_coloring(
            raft::handle_t const&                                             handle,
            rocgraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
            raft::random::RngState&                                           rng_state)
        {
            using graph_view_t = rocgraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>;
            graph_view_t current_graph_view(graph_view);

            // edge mask
            /* edge_property_bool */
            rocgraph::edge_property_t<graph_view_t, bool> edge_masks_even(handle,
                                                                          current_graph_view);
            rocgraph::fill_edge_property(handle, current_graph_view, bool{false}, edge_masks_even);

            /* edge_property_bool */
            rocgraph::edge_property_t<graph_view_t, bool> edge_masks_odd(handle,
                                                                         current_graph_view);
            rocgraph::fill_edge_property(handle, current_graph_view, bool{false}, edge_masks_odd);

            rocgraph::transform_e(
                handle,
                current_graph_view,
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                rocgraph::edge_dummy_property_t{}.view(),
                [] __device__(
                    auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) {
                    return !(src == dst); // mask out self-loop
                },
                edge_masks_even.mutable_view());

            current_graph_view.attach_edge_mask(edge_masks_even.view());

            // device vector to store colors of vertices
            rmm::device_uvector<vertex_t> colors = rmm::device_uvector<vertex_t>(
                current_graph_view.local_vertex_partition_range_size(), handle.get_stream());
            thrust::fill(handle.get_thrust_policy(),
                         colors.begin(),
                         colors.end(),
                         std::numeric_limits<vertex_t>::max());

            vertex_t color_id = 0;
            while(true)
            {
                auto mis = rocgraph::maximal_independent_set<vertex_t, edge_t, multi_gpu>(
                    handle, current_graph_view, rng_state);

                using flag_t                                 = uint8_t;
                rmm::device_uvector<flag_t> is_vertex_in_mis = rmm::device_uvector<flag_t>(
                    current_graph_view.local_vertex_partition_range_size(), handle.get_stream());
                thrust::fill(handle.get_thrust_policy(),
                             is_vertex_in_mis.begin(),
                             is_vertex_in_mis.end(),
                             0);

                thrust::for_each(
                    handle.get_thrust_policy(),
                    mis.begin(),
                    mis.end(),
                    [color_id,
                     colors           = colors.data(),
                     is_vertex_in_mis = is_vertex_in_mis.data(),
                     v_first          = current_graph_view
                                   .local_vertex_partition_range_first()] __device__(vertex_t v) {
                        auto v_offset              = v - v_first;
                        is_vertex_in_mis[v_offset] = flag_t{1};
                        vertex_t initial_color_id  = colors[v_offset];
                        colors[v_offset]
                            = (color_id < initial_color_id) ? color_id : initial_color_id;
                    });

                if(current_graph_view.compute_number_of_edges(handle) == 0)
                {
                    break;
                }

                rocgraph::edge_src_property_t<graph_view_t, flag_t> src_mis_flags(handle);
                rocgraph::edge_dst_property_t<graph_view_t, flag_t> dst_mis_flags(handle);

                if constexpr(graph_view_t::is_multi_gpu)
                {
                    src_mis_flags = rocgraph::edge_src_property_t<graph_view_t, flag_t>(
                        handle, current_graph_view);
                    dst_mis_flags = rocgraph::edge_dst_property_t<graph_view_t, flag_t>(
                        handle, current_graph_view);

                    rocgraph::update_edge_src_property(
                        handle, current_graph_view, is_vertex_in_mis.begin(), src_mis_flags);

                    rocgraph::update_edge_dst_property(
                        handle, current_graph_view, is_vertex_in_mis.begin(), dst_mis_flags);
                }

                if(color_id % 2 == 0)
                {
                    rocgraph::transform_e(
                        handle,
                        current_graph_view,
                        graph_view_t::is_multi_gpu
                            ? src_mis_flags.view()
                            : detail::edge_major_property_view_t<vertex_t, flag_t const*>(
                                  is_vertex_in_mis.begin()),
                        graph_view_t::is_multi_gpu
                            ? dst_mis_flags.view()
                            : detail::edge_minor_property_view_t<vertex_t, flag_t const*>(
                                  is_vertex_in_mis.begin(), vertex_t{0}),
                        rocgraph::edge_dummy_property_t{}.view(),
                        [color_id] __device__(auto src,
                                              auto dst,
                                              auto is_src_in_mis,
                                              auto is_dst_in_mis,
                                              thrust::nullopt_t) {
                            return !((is_src_in_mis == uint8_t{true})
                                     || (is_dst_in_mis == uint8_t{true}));
                        },
                        edge_masks_odd.mutable_view());

                    if(current_graph_view.has_edge_mask())
                        current_graph_view.clear_edge_mask();
                    rocgraph::fill_edge_property(
                        handle, current_graph_view, bool{false}, edge_masks_even);
                    current_graph_view.attach_edge_mask(edge_masks_odd.view());
                }
                else
                {
                    rocgraph::transform_e(
                        handle,
                        current_graph_view,
                        graph_view_t::is_multi_gpu
                            ? src_mis_flags.view()
                            : detail::edge_major_property_view_t<vertex_t, flag_t const*>(
                                  is_vertex_in_mis.begin()),
                        graph_view_t::is_multi_gpu
                            ? dst_mis_flags.view()
                            : detail::edge_minor_property_view_t<vertex_t, flag_t const*>(
                                  is_vertex_in_mis.begin(), vertex_t{0}),
                        rocgraph::edge_dummy_property_t{}.view(),
                        [color_id] __device__(auto src,
                                              auto dst,
                                              auto is_src_in_mis,
                                              auto is_dst_in_mis,
                                              thrust::nullopt_t) {
                            return !((is_src_in_mis == uint8_t{true})
                                     || (is_dst_in_mis == uint8_t{true}));
                        },
                        edge_masks_even.mutable_view());

                    if(current_graph_view.has_edge_mask())
                        current_graph_view.clear_edge_mask();
                    rocgraph::fill_edge_property(
                        handle, current_graph_view, bool{false}, edge_masks_odd);
                    current_graph_view.attach_edge_mask(edge_masks_even.view());
                }

                color_id++;
            }
            return colors;
        }
    } // namespace detail

    template <typename vertex_t, typename edge_t, bool multi_gpu>
    rmm::device_uvector<vertex_t>
        vertex_coloring(raft::handle_t const&                                   handle,
                        graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                        raft::random::RngState&                                 rng_state)
    {
        return detail::vertex_coloring(handle, graph_view, rng_state);
    }

} // namespace rocgraph
