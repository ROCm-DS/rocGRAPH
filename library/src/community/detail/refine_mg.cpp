// Copyright (C) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "community/detail/refine_impl.cuh"

namespace rocgraph
{
    namespace detail
    {

        template std::tuple<rmm::device_uvector<int32_t>,
                            std::pair<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>>
            refine_clustering(
                raft::handle_t const&                                        handle,
                raft::random::RngState&                                      rng_state,
                rocgraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
                std::optional<edge_property_view_t<int32_t, float const*>>   edge_weight_view,
                float                                                        total_edge_weight,
                float                                                        resolution,
                float                                                        theta,
                rmm::device_uvector<float> const&                            vertex_weights_v,
                rmm::device_uvector<int32_t>&&                               cluster_keys_v,
                rmm::device_uvector<float>&&                                 cluster_weights_v,
                rmm::device_uvector<int32_t>&&                               next_clusters_v,
                edge_src_property_t<rocgraph::graph_view_t<int32_t, int32_t, false, true>,
                                    float> const&   src_vertex_weights_cache,
                edge_src_property_t<rocgraph::graph_view_t<int32_t, int32_t, false, true>,
                                    int32_t> const& src_clusters_cache,
                edge_dst_property_t<rocgraph::graph_view_t<int32_t, int32_t, false, true>,
                                    int32_t> const& dst_clusters_cache,
                bool                                up_down);

        template std::tuple<rmm::device_uvector<int32_t>,
                            std::pair<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>>
            refine_clustering(
                raft::handle_t const&                                        handle,
                raft::random::RngState&                                      rng_state,
                rocgraph::graph_view_t<int32_t, int64_t, false, true> const& graph_view,
                std::optional<edge_property_view_t<int64_t, float const*>>   edge_weight_view,
                float                                                        total_edge_weight,
                float                                                        resolution,
                float                                                        theta,
                rmm::device_uvector<float> const&                            vertex_weights_v,
                rmm::device_uvector<int32_t>&&                               cluster_keys_v,
                rmm::device_uvector<float>&&                                 cluster_weights_v,
                rmm::device_uvector<int32_t>&&                               next_clusters_v,
                edge_src_property_t<rocgraph::graph_view_t<int32_t, int64_t, false, true>,
                                    float> const&   src_vertex_weights_cache,
                edge_src_property_t<rocgraph::graph_view_t<int32_t, int64_t, false, true>,
                                    int32_t> const& src_clusters_cache,
                edge_dst_property_t<rocgraph::graph_view_t<int32_t, int64_t, false, true>,
                                    int32_t> const& dst_clusters_cache,
                bool                                up_down);

        template std::tuple<rmm::device_uvector<int64_t>,
                            std::pair<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>>
            refine_clustering(
                raft::handle_t const&                                        handle,
                raft::random::RngState&                                      rng_state,
                rocgraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                std::optional<edge_property_view_t<int64_t, float const*>>   edge_weight_view,
                float                                                        total_edge_weight,
                float                                                        resolution,
                float                                                        theta,
                rmm::device_uvector<float> const&                            vertex_weights_v,
                rmm::device_uvector<int64_t>&&                               cluster_keys_v,
                rmm::device_uvector<float>&&                                 cluster_weights_v,
                rmm::device_uvector<int64_t>&&                               next_clusters_v,
                edge_src_property_t<rocgraph::graph_view_t<int64_t, int64_t, false, true>,
                                    float> const&   src_vertex_weights_cache,
                edge_src_property_t<rocgraph::graph_view_t<int64_t, int64_t, false, true>,
                                    int64_t> const& src_clusters_cache,
                edge_dst_property_t<rocgraph::graph_view_t<int64_t, int64_t, false, true>,
                                    int64_t> const& dst_clusters_cache,
                bool                                up_down);

        template std::tuple<rmm::device_uvector<int32_t>,
                            std::pair<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>>
            refine_clustering(
                raft::handle_t const&                                        handle,
                raft::random::RngState&                                      rng_state,
                rocgraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
                std::optional<edge_property_view_t<int32_t, double const*>>  edge_weight_view,
                double                                                       total_edge_weight,
                double                                                       resolution,
                double                                                       theta,
                rmm::device_uvector<double> const&                           vertex_weights_v,
                rmm::device_uvector<int32_t>&&                               cluster_keys_v,
                rmm::device_uvector<double>&&                                cluster_weights_v,
                rmm::device_uvector<int32_t>&&                               next_clusters_v,
                edge_src_property_t<rocgraph::graph_view_t<int32_t, int32_t, false, true>,
                                    double> const&  src_vertex_weights_cache,
                edge_src_property_t<rocgraph::graph_view_t<int32_t, int32_t, false, true>,
                                    int32_t> const& src_clusters_cache,
                edge_dst_property_t<rocgraph::graph_view_t<int32_t, int32_t, false, true>,
                                    int32_t> const& dst_clusters_cache,
                bool                                up_down);

        template std::tuple<rmm::device_uvector<int32_t>,
                            std::pair<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>>
            refine_clustering(
                raft::handle_t const&                                        handle,
                raft::random::RngState&                                      rng_state,
                rocgraph::graph_view_t<int32_t, int64_t, false, true> const& graph_view,
                std::optional<edge_property_view_t<int64_t, double const*>>  edge_weight_view,
                double                                                       total_edge_weight,
                double                                                       resolution,
                double                                                       theta,
                rmm::device_uvector<double> const&                           vertex_weights_v,
                rmm::device_uvector<int32_t>&&                               cluster_keys_v,
                rmm::device_uvector<double>&&                                cluster_weights_v,
                rmm::device_uvector<int32_t>&&                               next_clusters_v,
                edge_src_property_t<rocgraph::graph_view_t<int32_t, int64_t, false, true>,
                                    double> const&  src_vertex_weights_cache,
                edge_src_property_t<rocgraph::graph_view_t<int32_t, int64_t, false, true>,
                                    int32_t> const& src_clusters_cache,
                edge_dst_property_t<rocgraph::graph_view_t<int32_t, int64_t, false, true>,
                                    int32_t> const& dst_clusters_cache,
                bool                                up_down);

        template std::tuple<rmm::device_uvector<int64_t>,
                            std::pair<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>>
            refine_clustering(
                raft::handle_t const&                                        handle,
                raft::random::RngState&                                      rng_state,
                rocgraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                std::optional<edge_property_view_t<int64_t, double const*>>  edge_weight_view,
                double                                                       total_edge_weight,
                double                                                       resolution,
                double                                                       theta,
                rmm::device_uvector<double> const&                           vertex_weights_v,
                rmm::device_uvector<int64_t>&&                               cluster_keys_v,
                rmm::device_uvector<double>&&                                cluster_weights_v,
                rmm::device_uvector<int64_t>&&                               next_clusters_v,
                edge_src_property_t<rocgraph::graph_view_t<int64_t, int64_t, false, true>,
                                    double> const&  src_vertex_weights_cache,
                edge_src_property_t<rocgraph::graph_view_t<int64_t, int64_t, false, true>,
                                    int64_t> const& src_clusters_cache,
                edge_dst_property_t<rocgraph::graph_view_t<int64_t, int64_t, false, true>,
                                    int64_t> const& dst_clusters_cache,
                bool                                up_down);

    } // namespace detail
} // namespace rocgraph
