// Copyright (C) 2020-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

// #define TIMING

// FIXME: Only outstanding items preventing this becoming a .hpp file
#include "community/detail/common_methods.hpp"
#include "community/flatten_dendrogram.hpp"
#include "prims/update_edge_src_dst_property.cuh"

#include "detail/collect_comm_wrapper.hpp"
#include "detail/shuffle_wrappers.hpp"
#include "detail/utility_wrappers.hpp"
#include "graph.hpp"
#include "graph_functions.hpp"

#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

namespace rocgraph
{

    namespace detail
    {

        template <typename vertex_t, typename edge_t, bool multi_gpu>
        void check_clustering(graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                              vertex_t*                                               clustering)
        {
            if(graph_view.local_vertex_partition_range_size() > 0)
                ROCGRAPH_EXPECTS(clustering != nullptr,
                                 "Invalid input argument: clustering is null");
        }

        template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
        std::pair<std::unique_ptr<Dendrogram<vertex_t>>, weight_t>
            louvain(raft::handle_t const&                                         handle,
                    std::optional<std::reference_wrapper<raft::random::RngState>> rng_state,
                    graph_view_t<vertex_t, edge_t, false, multi_gpu> const&       graph_view,
                    std::optional<edge_property_view_t<edge_t, weight_t const*>>  edge_weight_view,
                    size_t                                                        max_level,
                    weight_t                                                      threshold,
                    weight_t                                                      resolution)
        {
            using graph_t      = rocgraph::graph_t<vertex_t, edge_t, false, multi_gpu>;
            using graph_view_t = rocgraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>;

            ROCGRAPH_EXPECTS(edge_weight_view.has_value(), "Graph must be weighted");

            std::unique_ptr<Dendrogram<vertex_t>> dendrogram
                = std::make_unique<Dendrogram<vertex_t>>();
            graph_t                                                current_graph(handle);
            graph_view_t                                           current_graph_view(graph_view);
            std::optional<edge_property_t<graph_view_t, weight_t>> current_edge_weights(handle);
            std::optional<edge_property_view_t<edge_t, weight_t const*>> current_edge_weight_view(
                edge_weight_view);

            weight_t best_modularity = weight_t{-1};
            weight_t total_edge_weight
                = compute_total_edge_weight(handle, current_graph_view, *current_edge_weight_view);

            rmm::device_uvector<vertex_t>               cluster_keys_v(0, handle.get_stream());
            rmm::device_uvector<weight_t>               cluster_weights_v(0, handle.get_stream());
            rmm::device_uvector<weight_t>               vertex_weights_v(0, handle.get_stream());
            rmm::device_uvector<vertex_t>               next_clusters_v(0, handle.get_stream());
            edge_src_property_t<graph_view_t, weight_t> src_vertex_weights_cache(handle);
            edge_src_property_t<graph_view_t, vertex_t> src_clusters_cache(handle);
            edge_dst_property_t<graph_view_t, vertex_t> dst_clusters_cache(handle);

            while(dendrogram->num_levels() < max_level)
            {
                //
                //  Initialize every cluster to reference each vertex to itself
                //
                dendrogram->add_level(current_graph_view.local_vertex_partition_range_first(),
                                      current_graph_view.local_vertex_partition_range_size(),
                                      handle.get_stream());

                if(rng_state)
                {
                    auto random_cluster_assignments = rocgraph::detail::permute_range<vertex_t>(
                        handle,
                        *rng_state,
                        current_graph_view.local_vertex_partition_range_first(),
                        current_graph_view.local_vertex_partition_range_size(),
                        multi_gpu);

                    raft::copy(dendrogram->current_level_begin(),
                               random_cluster_assignments.begin(),
                               random_cluster_assignments.size(),
                               handle.get_stream());
                }
                else
                {
                    detail::sequence_fill(handle.get_stream(),
                                          dendrogram->current_level_begin(),
                                          dendrogram->current_level_size(),
                                          current_graph_view.local_vertex_partition_range_first());
                }
                //
                //  Compute the vertex and cluster weights, these are different for each
                //  graph in the hierarchical decomposition
                //

#ifdef TIMING
                detail::timer_start<graph_view_t::is_multi_gpu>(
                    handle, hr_timer, "compute_vertex_and_cluster_weights");
#endif

                vertex_weights_v = compute_out_weight_sums(
                    handle, current_graph_view, *current_edge_weight_view);
                cluster_keys_v.resize(vertex_weights_v.size(), handle.get_stream());
                cluster_weights_v.resize(vertex_weights_v.size(), handle.get_stream());

                detail::sequence_fill(handle.get_stream(),
                                      cluster_keys_v.begin(),
                                      cluster_keys_v.size(),
                                      current_graph_view.local_vertex_partition_range_first());

                raft::copy(cluster_weights_v.begin(),
                           vertex_weights_v.begin(),
                           vertex_weights_v.size(),
                           handle.get_stream());

                if constexpr(graph_view_t::is_multi_gpu)
                {
                    std::tie(cluster_keys_v, cluster_weights_v) = detail::
                        shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
                            handle, std::move(cluster_keys_v), std::move(cluster_weights_v));

                    src_vertex_weights_cache
                        = edge_src_property_t<graph_view_t, weight_t>(handle, current_graph_view);
                    update_edge_src_property(handle,
                                             current_graph_view,
                                             vertex_weights_v.begin(),
                                             src_vertex_weights_cache);
                    vertex_weights_v.resize(0, handle.get_stream());
                    vertex_weights_v.shrink_to_fit(handle.get_stream());
                }

#ifdef TIMING
                detail::timer_stop<graph_view_t::is_multi_gpu>(handle, hr_timer);
#endif

                //
                //  Update the clustering assignment, this is the main loop of Louvain
                //

#ifdef TIMING
                detail::timer_start<graph_view_t::is_multi_gpu>(
                    handle, hr_timer, "update_clustering");
#endif

                next_clusters_v = rmm::device_uvector<vertex_t>(dendrogram->current_level_size(),
                                                                handle.get_stream());

                raft::copy(next_clusters_v.begin(),
                           dendrogram->current_level_begin(),
                           dendrogram->current_level_size(),
                           handle.get_stream());

                if constexpr(multi_gpu)
                {
                    src_clusters_cache
                        = edge_src_property_t<graph_view_t, vertex_t>(handle, current_graph_view);
                    update_edge_src_property(
                        handle, current_graph_view, next_clusters_v.begin(), src_clusters_cache);
                    dst_clusters_cache
                        = edge_dst_property_t<graph_view_t, vertex_t>(handle, current_graph_view);
                    update_edge_dst_property(
                        handle, current_graph_view, next_clusters_v.begin(), dst_clusters_cache);
                }

                weight_t new_Q = detail::compute_modularity(handle,
                                                            current_graph_view,
                                                            current_edge_weight_view,
                                                            src_clusters_cache,
                                                            dst_clusters_cache,
                                                            next_clusters_v,
                                                            cluster_weights_v,
                                                            total_edge_weight,
                                                            resolution);
                weight_t cur_Q = new_Q - 1;

                // To avoid the potential of having two vertices swap clusters
                // we will only allow vertices to move up (true) or down (false)
                // during each iteration of the loop
                bool up_down = true;

                while(new_Q > (cur_Q + threshold))
                {
                    cur_Q = new_Q;

                    next_clusters_v = detail::update_clustering_by_delta_modularity(
                        handle,
                        current_graph_view,
                        current_edge_weight_view,
                        total_edge_weight,
                        resolution,
                        vertex_weights_v,
                        std::move(cluster_keys_v),
                        std::move(cluster_weights_v),
                        std::move(next_clusters_v),
                        src_vertex_weights_cache,
                        src_clusters_cache,
                        dst_clusters_cache,
                        up_down);

                    if constexpr(graph_view_t::is_multi_gpu)
                    {
                        update_edge_src_property(handle,
                                                 current_graph_view,
                                                 next_clusters_v.begin(),
                                                 src_clusters_cache);
                        update_edge_dst_property(handle,
                                                 current_graph_view,
                                                 next_clusters_v.begin(),
                                                 dst_clusters_cache);
                    }

                    std::tie(cluster_keys_v, cluster_weights_v)
                        = detail::compute_cluster_keys_and_values(handle,
                                                                  current_graph_view,
                                                                  current_edge_weight_view,
                                                                  next_clusters_v,
                                                                  src_clusters_cache);

                    up_down = !up_down;

                    new_Q = detail::compute_modularity(handle,
                                                       current_graph_view,
                                                       current_edge_weight_view,
                                                       src_clusters_cache,
                                                       dst_clusters_cache,
                                                       next_clusters_v,
                                                       cluster_weights_v,
                                                       total_edge_weight,
                                                       resolution);

                    if(new_Q > cur_Q)
                    {
                        raft::copy(dendrogram->current_level_begin(),
                                   next_clusters_v.begin(),
                                   next_clusters_v.size(),
                                   handle.get_stream());
                    }
                }

#ifdef TIMING
                detail::timer_stop<graph_view_t::is_multi_gpu>(handle, hr_timer);
#endif

                if(cur_Q <= best_modularity)
                {
                    break;
                }

                best_modularity = cur_Q;

                //
                //  Contract the graph
                //

#ifdef TIMING
                detail::timer_start<graph_view_t::is_multi_gpu>(handle, hr_timer, "contract graph");
#endif

                cluster_keys_v.resize(0, handle.get_stream());
                cluster_weights_v.resize(0, handle.get_stream());
                vertex_weights_v.resize(0, handle.get_stream());
                next_clusters_v.resize(0, handle.get_stream());
                cluster_keys_v.shrink_to_fit(handle.get_stream());
                cluster_weights_v.shrink_to_fit(handle.get_stream());
                vertex_weights_v.shrink_to_fit(handle.get_stream());
                next_clusters_v.shrink_to_fit(handle.get_stream());
                src_vertex_weights_cache.clear(handle);
                src_clusters_cache.clear(handle);
                dst_clusters_cache.clear(handle);

                std::tie(current_graph, current_edge_weights) = rocgraph::detail::graph_contraction(
                    handle,
                    current_graph_view,
                    current_edge_weight_view,
                    raft::device_span<vertex_t>{dendrogram->current_level_begin(),
                                                dendrogram->current_level_size()});
                current_graph_view = current_graph.view();
                current_edge_weight_view
                    = std::make_optional<edge_property_view_t<edge_t, weight_t const*>>(
                        (*current_edge_weights).view());

#ifdef TIMING
                detail::timer_stop<graph_view_t::is_multi_gpu>(handle, hr_timer);
#endif
            }

#ifdef TIMING
            detail::timer_display<graph_view_t::is_multi_gpu>(handle, hr_timer, std::cout);
#endif

            return std::make_pair(std::move(dendrogram), best_modularity);
        }

        template <typename vertex_t, typename edge_t, bool multi_gpu>
        void flatten_dendrogram(raft::handle_t const&                                   handle,
                                graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                                Dendrogram<vertex_t> const&                             dendrogram,
                                vertex_t*                                               clustering)
        {
            rmm::device_uvector<vertex_t> vertex_ids_v(graph_view.number_of_vertices(),
                                                       handle.get_stream());

            detail::sequence_fill(handle.get_stream(),
                                  vertex_ids_v.begin(),
                                  vertex_ids_v.size(),
                                  graph_view.local_vertex_partition_range_first());

            partition_at_level<vertex_t, multi_gpu>(
                handle, dendrogram, vertex_ids_v.data(), clustering, dendrogram.num_levels());
        }

    } // namespace detail

    template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
    std::pair<std::unique_ptr<Dendrogram<vertex_t>>, weight_t>
        louvain(raft::handle_t const&                                         handle,
                std::optional<std::reference_wrapper<raft::random::RngState>> rng_state,
                graph_view_t<vertex_t, edge_t, false, multi_gpu> const&       graph_view,
                std::optional<edge_property_view_t<edge_t, weight_t const*>>  edge_weight_view,
                size_t                                                        max_level,
                weight_t                                                      threshold,
                weight_t                                                      resolution)
    {
        ROCGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

        ROCGRAPH_EXPECTS(edge_weight_view.has_value(), "Graph must be weighted");

        return detail::louvain(
            handle, rng_state, graph_view, edge_weight_view, max_level, threshold, resolution);
    }

    template <typename vertex_t, typename edge_t, bool multi_gpu>
    void flatten_dendrogram(raft::handle_t const&                                   handle,
                            graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                            Dendrogram<vertex_t> const&                             dendrogram,
                            vertex_t*                                               clustering)
    {
        ROCGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

        detail::flatten_dendrogram(handle, graph_view, dendrogram, clustering);
    }

    template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
    std::pair<size_t, weight_t>
        louvain(raft::handle_t const&                                         handle,
                std::optional<std::reference_wrapper<raft::random::RngState>> rng_state,
                graph_view_t<vertex_t, edge_t, false, multi_gpu> const&       graph_view,
                std::optional<edge_property_view_t<edge_t, weight_t const*>>  edge_weight_view,
                vertex_t*                                                     clustering,
                size_t                                                        max_level,
                weight_t                                                      threshold,
                weight_t                                                      resolution)
    {
        ROCGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

        ROCGRAPH_EXPECTS(edge_weight_view.has_value(), "Graph must be weighted");
        detail::check_clustering(graph_view, clustering);

        std::unique_ptr<Dendrogram<vertex_t>> dendrogram;
        weight_t                              modularity;

        std::tie(dendrogram, modularity) = detail::louvain(
            handle, rng_state, graph_view, edge_weight_view, max_level, threshold, resolution);

        detail::flatten_dendrogram(handle, graph_view, *dendrogram, clustering);

        return std::make_pair(dendrogram->num_levels(), modularity);
    }

} // namespace rocgraph
