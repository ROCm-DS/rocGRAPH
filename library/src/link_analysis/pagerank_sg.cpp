// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "link_analysis/pagerank_impl.cuh"

namespace rocgraph
{

    // SG instantiation
    template void
        pagerank(raft::handle_t const&                                      handle,
                 graph_view_t<int32_t, int32_t, true, false> const&         graph_view,
                 std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
                 std::optional<float const*>   precomputed_vertex_out_weight_sums,
                 std::optional<int32_t const*> personalization_vertices,
                 std::optional<float const*>   personalization_values,
                 std::optional<int32_t>        personalization_vector_size,
                 float*                        pageranks,
                 float                         alpha,
                 float                         epsilon,
                 size_t                        max_iterations,
                 bool                          has_initial_guess,
                 bool                          do_expensive_check);

    template void
        pagerank(raft::handle_t const&                                       handle,
                 graph_view_t<int32_t, int32_t, true, false> const&          graph_view,
                 std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
                 std::optional<double const*>  precomputed_vertex_out_weight_sums,
                 std::optional<int32_t const*> personalization_vertices,
                 std::optional<double const*>  personalization_values,
                 std::optional<int32_t>        personalization_vector_size,
                 double*                       pageranks,
                 double                        alpha,
                 double                        epsilon,
                 size_t                        max_iterations,
                 bool                          has_initial_guess,
                 bool                          do_expensive_check);

    template void
        pagerank(raft::handle_t const&                                      handle,
                 graph_view_t<int32_t, int64_t, true, false> const&         graph_view,
                 std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
                 std::optional<float const*>   precomputed_vertex_out_weight_sums,
                 std::optional<int32_t const*> personalization_vertices,
                 std::optional<float const*>   personalization_values,
                 std::optional<int32_t>        personalization_vector_size,
                 float*                        pageranks,
                 float                         alpha,
                 float                         epsilon,
                 size_t                        max_iterations,
                 bool                          has_initial_guess,
                 bool                          do_expensive_check);

    template void
        pagerank(raft::handle_t const&                                       handle,
                 graph_view_t<int32_t, int64_t, true, false> const&          graph_view,
                 std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
                 std::optional<double const*>  precomputed_vertex_out_weight_sums,
                 std::optional<int32_t const*> personalization_vertices,
                 std::optional<double const*>  personalization_values,
                 std::optional<int32_t>        personalization_vector_size,
                 double*                       pageranks,
                 double                        alpha,
                 double                        epsilon,
                 size_t                        max_iterations,
                 bool                          has_initial_guess,
                 bool                          do_expensive_check);

    template void
        pagerank(raft::handle_t const&                                      handle,
                 graph_view_t<int64_t, int64_t, true, false> const&         graph_view,
                 std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
                 std::optional<float const*>   precomputed_vertex_out_weight_sums,
                 std::optional<int64_t const*> personalization_vertices,
                 std::optional<float const*>   personalization_values,
                 std::optional<int64_t>        personalization_vector_size,
                 float*                        pageranks,
                 float                         alpha,
                 float                         epsilon,
                 size_t                        max_iterations,
                 bool                          has_initial_guess,
                 bool                          do_expensive_check);

    template void
        pagerank(raft::handle_t const&                                       handle,
                 graph_view_t<int64_t, int64_t, true, false> const&          graph_view,
                 std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
                 std::optional<double const*>  precomputed_vertex_out_weight_sums,
                 std::optional<int64_t const*> personalization_vertices,
                 std::optional<double const*>  personalization_values,
                 std::optional<int64_t>        personalization_vector_size,
                 double*                       pageranks,
                 double                        alpha,
                 double                        epsilon,
                 size_t                        max_iterations,
                 bool                          has_initial_guess,
                 bool                          do_expensive_check);

    template std::tuple<rmm::device_uvector<float>, centrality_algorithm_metadata_t> pagerank(
        raft::handle_t const&                                      handle,
        graph_view_t<int32_t, int32_t, true, false> const&         graph_view,
        std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
        std::optional<raft::device_span<float const>> precomputed_vertex_out_weight_sums,
        std::optional<std::tuple<raft::device_span<int32_t const>, raft::device_span<float const>>>
                                                      personalization,
        std::optional<raft::device_span<float const>> initial_pageranks,
        float                                         alpha,
        float                                         epsilon,
        size_t                                        max_iterations,
        bool                                          do_expensive_check);

    template std::tuple<rmm::device_uvector<double>, centrality_algorithm_metadata_t> pagerank(
        raft::handle_t const&                                       handle,
        graph_view_t<int32_t, int32_t, true, false> const&          graph_view,
        std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
        std::optional<raft::device_span<double const>> precomputed_vertex_out_weight_sums,
        std::optional<std::tuple<raft::device_span<int32_t const>, raft::device_span<double const>>>
                                                       personalization,
        std::optional<raft::device_span<double const>> initial_pageranks,
        double                                         alpha,
        double                                         epsilon,
        size_t                                         max_iterations,
        bool                                           do_expensive_check);

    template std::tuple<rmm::device_uvector<float>, centrality_algorithm_metadata_t> pagerank(
        raft::handle_t const&                                      handle,
        graph_view_t<int32_t, int64_t, true, false> const&         graph_view,
        std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
        std::optional<raft::device_span<float const>> precomputed_vertex_out_weight_sums,
        std::optional<std::tuple<raft::device_span<int32_t const>, raft::device_span<float const>>>
                                                      personalization,
        std::optional<raft::device_span<float const>> initial_pageranks,
        float                                         alpha,
        float                                         epsilon,
        size_t                                        max_iterations,
        bool                                          do_expensive_check);

    template std::tuple<rmm::device_uvector<double>, centrality_algorithm_metadata_t> pagerank(
        raft::handle_t const&                                       handle,
        graph_view_t<int32_t, int64_t, true, false> const&          graph_view,
        std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
        std::optional<raft::device_span<double const>> precomputed_vertex_out_weight_sums,
        std::optional<std::tuple<raft::device_span<int32_t const>, raft::device_span<double const>>>
                                                       personalization,
        std::optional<raft::device_span<double const>> initial_pageranks,
        double                                         alpha,
        double                                         epsilon,
        size_t                                         max_iterations,
        bool                                           do_expensive_check);

    template std::tuple<rmm::device_uvector<float>, centrality_algorithm_metadata_t> pagerank(
        raft::handle_t const&                                      handle,
        graph_view_t<int64_t, int64_t, true, false> const&         graph_view,
        std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
        std::optional<raft::device_span<float const>> precomputed_vertex_out_weight_sums,
        std::optional<std::tuple<raft::device_span<int64_t const>, raft::device_span<float const>>>
                                                      personalization,
        std::optional<raft::device_span<float const>> initial_pageranks,
        float                                         alpha,
        float                                         epsilon,
        size_t                                        max_iterations,
        bool                                          do_expensive_check);

    template std::tuple<rmm::device_uvector<double>, centrality_algorithm_metadata_t> pagerank(
        raft::handle_t const&                                       handle,
        graph_view_t<int64_t, int64_t, true, false> const&          graph_view,
        std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
        std::optional<raft::device_span<double const>> precomputed_vertex_out_weight_sums,
        std::optional<std::tuple<raft::device_span<int64_t const>, raft::device_span<double const>>>
                                                       personalization,
        std::optional<raft::device_span<double const>> initial_pageranks,
        double                                         alpha,
        double                                         epsilon,
        size_t                                         max_iterations,
        bool                                           do_expensive_check);

} // namespace rocgraph
