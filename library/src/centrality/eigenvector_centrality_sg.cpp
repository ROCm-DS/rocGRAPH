// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "centrality/eigenvector_centrality_impl.cuh"

namespace rocgraph
{

    // SG instantiation

    template rmm::device_uvector<float> eigenvector_centrality(
        raft::handle_t const&                                      handle,
        graph_view_t<int32_t, int32_t, true, false> const&         graph_view,
        std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
        std::optional<raft::device_span<float const>>              initial_centralities,
        float                                                      epsilon,
        size_t                                                     max_iterations,
        bool                                                       do_expensive_check);

    template rmm::device_uvector<float> eigenvector_centrality(
        raft::handle_t const&                                      handle,
        graph_view_t<int32_t, int64_t, true, false> const&         graph_view,
        std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
        std::optional<raft::device_span<float const>>              initial_centralities,
        float                                                      epsilon,
        size_t                                                     max_iterations,
        bool                                                       do_expensive_check);

    template rmm::device_uvector<float> eigenvector_centrality(
        raft::handle_t const&                                      handle,
        graph_view_t<int64_t, int64_t, true, false> const&         graph_view,
        std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
        std::optional<raft::device_span<float const>>              initial_centralities,
        float                                                      epsilon,
        size_t                                                     max_iterations,
        bool                                                       do_expensive_check);

    template rmm::device_uvector<double> eigenvector_centrality(
        raft::handle_t const&                                       handle,
        graph_view_t<int32_t, int32_t, true, false> const&          graph_view,
        std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
        std::optional<raft::device_span<double const>>              initial_centralities,
        double                                                      epsilon,
        size_t                                                      max_iterations,
        bool                                                        do_expensive_check);

    template rmm::device_uvector<double> eigenvector_centrality(
        raft::handle_t const&                                       handle,
        graph_view_t<int32_t, int64_t, true, false> const&          graph_view,
        std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
        std::optional<raft::device_span<double const>>              initial_centralities,
        double                                                      epsilon,
        size_t                                                      max_iterations,
        bool                                                        do_expensive_check);

    template rmm::device_uvector<double> eigenvector_centrality(
        raft::handle_t const&                                       handle,
        graph_view_t<int64_t, int64_t, true, false> const&          graph_view,
        std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
        std::optional<raft::device_span<double const>>              initial_centralities,
        double                                                      epsilon,
        size_t                                                      max_iterations,
        bool                                                        do_expensive_check);

} // namespace rocgraph
