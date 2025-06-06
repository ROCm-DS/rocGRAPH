// Copyright (C) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "community/detail/common_methods.hpp"
#include "prims/fill_edge_property.cuh"
#include "prims/transform_e.cuh"
#include "prims/update_edge_src_dst_property.cuh"

#include "algorithms.hpp"
#include "edge_property.hpp"
#include "graph_functions.hpp"
#include "graph_view.hpp"

#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

namespace rocgraph
{

    namespace detail
    {

        template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
        std::tuple<rmm::device_uvector<vertex_t>, size_t, weight_t>
            ecg(raft::handle_t const&                                        handle,
                raft::random::RngState&                                      rng_state,
                graph_view_t<vertex_t, edge_t, false, multi_gpu> const&      graph_view,
                std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                weight_t                                                     min_weight,
                size_t                                                       ensemble_size,
                size_t                                                       max_level,
                weight_t                                                     threshold,
                weight_t                                                     resolution)
        {
            using graph_view_t = rocgraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>;

            ROCGRAPH_EXPECTS(min_weight >= weight_t{0.0},
                             "Invalid input arguments: min_weight must be positive");
            ROCGRAPH_EXPECTS(ensemble_size >= 1,
                             "Invalid input arguments: ensemble_size must be a non-zero integer");
            ROCGRAPH_EXPECTS(
                threshold > 0.0 && threshold <= 1.0,
                "Invalid input arguments: threshold must be a positive number in range (0.0, 1.0]");
            ROCGRAPH_EXPECTS(resolution > 0.0 && resolution <= 1.0,
                             "Invalid input arguments: resolution must be a positive number in "
                             "range (0.0, 1.0]");

            edge_src_property_t<graph_view_t, vertex_t> src_cluster_assignments(handle, graph_view);
            edge_dst_property_t<graph_view_t, vertex_t> dst_cluster_assignments(handle, graph_view);
            edge_property_t<graph_view_t, weight_t>     modified_edge_weights(handle, graph_view);

            rocgraph::fill_edge_property(handle, graph_view, weight_t{0}, modified_edge_weights);

            weight_t                      modularity = -1.0;
            rmm::device_uvector<vertex_t> cluster_assignments(
                graph_view.local_vertex_partition_range_size(), handle.get_stream());

            for(size_t i = 0; i < ensemble_size; i++)
            {
                std::tie(std::ignore, modularity) = rocgraph::louvain(
                    handle,
                    std::make_optional(std::reference_wrapper<raft::random::RngState>(rng_state)),
                    graph_view,
                    edge_weight_view,
                    cluster_assignments.data(),
                    size_t{1},
                    threshold,
                    resolution);

                rocgraph::update_edge_src_property(
                    handle, graph_view, cluster_assignments.begin(), src_cluster_assignments);
                rocgraph::update_edge_dst_property(
                    handle, graph_view, cluster_assignments.begin(), dst_cluster_assignments);

                rocgraph::transform_e(
                    handle,
                    graph_view,
                    src_cluster_assignments.view(),
                    dst_cluster_assignments.view(),
                    modified_edge_weights.view(),
                    [] __device__(
                        auto, auto, auto src_property, auto dst_property, auto edge_property) {
                        return edge_property + (src_property == dst_property);
                    },
                    modified_edge_weights.mutable_view());
            }

            rocgraph::transform_e(
                handle,
                graph_view,
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                view_concat(*edge_weight_view, modified_edge_weights.view()),
                [min_weight, ensemble_size = static_cast<weight_t>(ensemble_size)] __device__(
                    auto, auto, thrust::nullopt_t, thrust::nullopt_t, auto edge_properties) {
                    auto e_weight    = thrust::get<0>(edge_properties);
                    auto e_frequency = thrust::get<1>(edge_properties);
                    return min_weight + (e_weight - min_weight) * e_frequency / ensemble_size;
                },
                modified_edge_weights.mutable_view());

            std::tie(max_level, modularity) = rocgraph::louvain(
                handle,
                std::make_optional(std::reference_wrapper<raft::random::RngState>(rng_state)),
                graph_view,
                std::make_optional(modified_edge_weights.view()),
                cluster_assignments.data(),
                max_level,
                threshold,
                resolution);

            // Compute final modularity using original edge weights
            weight_t total_edge_weight
                = rocgraph::compute_total_edge_weight(handle, graph_view, *edge_weight_view);

            if constexpr(multi_gpu)
            {
                rocgraph::update_edge_src_property(
                    handle, graph_view, cluster_assignments.begin(), src_cluster_assignments);
                rocgraph::update_edge_dst_property(
                    handle, graph_view, cluster_assignments.begin(), dst_cluster_assignments);
            }

            auto [cluster_keys, cluster_weights]
                = rocgraph::detail::compute_cluster_keys_and_values(handle,
                                                                    graph_view,
                                                                    edge_weight_view,
                                                                    cluster_assignments,
                                                                    src_cluster_assignments);

            modularity = detail::compute_modularity(handle,
                                                    graph_view,
                                                    edge_weight_view,
                                                    src_cluster_assignments,
                                                    dst_cluster_assignments,
                                                    cluster_assignments,
                                                    cluster_weights,
                                                    total_edge_weight,
                                                    resolution);

            return std::make_tuple(std::move(cluster_assignments), max_level, modularity);
        }

    } // namespace detail

    template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
    std::tuple<rmm::device_uvector<vertex_t>, size_t, weight_t>
        ecg(raft::handle_t const&                                        handle,
            raft::random::RngState&                                      rng_state,
            graph_view_t<vertex_t, edge_t, false, multi_gpu> const&      graph_view,
            std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
            weight_t                                                     min_weight,
            size_t                                                       ensemble_size,
            size_t                                                       max_level,
            weight_t                                                     threshold,
            weight_t                                                     resolution)
    {
        return detail::ecg(handle,
                           rng_state,
                           graph_view,
                           edge_weight_view,
                           min_weight,
                           ensemble_size,
                           max_level,
                           threshold,
                           resolution);
    }

} // namespace rocgraph
