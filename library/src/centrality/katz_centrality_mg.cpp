// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "centrality/katz_centrality_impl.cuh"

namespace rocgraph
{

    // MG instantiation

    template void
        katz_centrality(raft::handle_t const&                                      handle,
                        graph_view_t<int32_t, int32_t, true, true> const&          graph_view,
                        std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
                        float const*                                               betas,
                        float* katz_centralities,
                        float  alpha,
                        float  beta,
                        float  epsilon,
                        size_t max_iterations,
                        bool   has_initial_guess,
                        bool   normalize,
                        bool   do_expensive_check);

    template void katz_centrality(
        raft::handle_t const&                                       handle,
        graph_view_t<int32_t, int32_t, true, true> const&           graph_view,
        std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
        double const*                                               betas,
        double*                                                     katz_centralities,
        double                                                      alpha,
        double                                                      beta,
        double                                                      epsilon,
        size_t                                                      max_iterations,
        bool                                                        has_initial_guess,
        bool                                                        normalize,
        bool                                                        do_expensive_check);

    template void
        katz_centrality(raft::handle_t const&                                      handle,
                        graph_view_t<int32_t, int64_t, true, true> const&          graph_view,
                        std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
                        float const*                                               betas,
                        float* katz_centralities,
                        float  alpha,
                        float  beta,
                        float  epsilon,
                        size_t max_iterations,
                        bool   has_initial_guess,
                        bool   normalize,
                        bool   do_expensive_check);

    template void katz_centrality(
        raft::handle_t const&                                       handle,
        graph_view_t<int32_t, int64_t, true, true> const&           graph_view,
        std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
        double const*                                               betas,
        double*                                                     katz_centralities,
        double                                                      alpha,
        double                                                      beta,
        double                                                      epsilon,
        size_t                                                      max_iterations,
        bool                                                        has_initial_guess,
        bool                                                        normalize,
        bool                                                        do_expensive_check);

    template void
        katz_centrality(raft::handle_t const&                                      handle,
                        graph_view_t<int64_t, int64_t, true, true> const&          graph_view,
                        std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
                        float const*                                               betas,
                        float* katz_centralities,
                        float  alpha,
                        float  beta,
                        float  epsilon,
                        size_t max_iterations,
                        bool   has_initial_guess,
                        bool   normalize,
                        bool   do_expensive_check);

    template void katz_centrality(
        raft::handle_t const&                                       handle,
        graph_view_t<int64_t, int64_t, true, true> const&           graph_view,
        std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
        double const*                                               betas,
        double*                                                     katz_centralities,
        double                                                      alpha,
        double                                                      beta,
        double                                                      epsilon,
        size_t                                                      max_iterations,
        bool                                                        has_initial_guess,
        bool                                                        normalize,
        bool                                                        do_expensive_check);

} // namespace rocgraph
