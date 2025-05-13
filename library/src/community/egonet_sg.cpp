// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "community/egonet_impl.cuh"

namespace rocgraph
{

    // SG FP32

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<float>>,
                        rmm::device_uvector<size_t>>
        extract_ego(raft::handle_t const&                               handle,
                    graph_view_t<int32_t, int32_t, false, false> const& graph_view,
                    std::optional<edge_property_view_t<int32_t, float const*>>,
                    int32_t* source_vertex,
                    int32_t  n_subgraphs,
                    int32_t  radius);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<float>>,
                        rmm::device_uvector<size_t>>
        extract_ego(raft::handle_t const&                               handle,
                    graph_view_t<int32_t, int64_t, false, false> const& graph_view,
                    std::optional<edge_property_view_t<int64_t, float const*>>,
                    int32_t* source_vertex,
                    int32_t  n_subgraphs,
                    int32_t  radius);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>,
                        std::optional<rmm::device_uvector<float>>,
                        rmm::device_uvector<size_t>>
        extract_ego(raft::handle_t const&                               handle,
                    graph_view_t<int64_t, int64_t, false, false> const& graph_view,
                    std::optional<edge_property_view_t<int64_t, float const*>>,
                    int64_t* source_vertex,
                    int64_t  n_subgraphs,
                    int64_t  radius);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<float>>,
                        rmm::device_uvector<size_t>>
        extract_ego(raft::handle_t const&                               handle,
                    graph_view_t<int32_t, int32_t, false, false> const& graph_view,
                    std::optional<edge_property_view_t<int32_t, float const*>>,
                    raft::device_span<int32_t const> source_vertex,
                    int32_t                          radius,
                    bool                             do_expensive_check);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<float>>,
                        rmm::device_uvector<size_t>>
        extract_ego(raft::handle_t const&                               handle,
                    graph_view_t<int32_t, int64_t, false, false> const& graph_view,
                    std::optional<edge_property_view_t<int64_t, float const*>>,
                    raft::device_span<int32_t const> source_vertex,
                    int32_t                          radius,
                    bool                             do_expensive_check);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>,
                        std::optional<rmm::device_uvector<float>>,
                        rmm::device_uvector<size_t>>
        extract_ego(raft::handle_t const&                               handle,
                    graph_view_t<int64_t, int64_t, false, false> const& graph_view,
                    std::optional<edge_property_view_t<int64_t, float const*>>,
                    raft::device_span<int64_t const> source_vertex,
                    int64_t                          radius,
                    bool                             do_expensive_check);

    // SG FP64

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<double>>,
                        rmm::device_uvector<size_t>>
        extract_ego(raft::handle_t const&,
                    graph_view_t<int32_t, int32_t, false, false> const& graph_view,
                    std::optional<edge_property_view_t<int32_t, double const*>>,
                    int32_t* source_vertex,
                    int32_t  n_subgraphs,
                    int32_t  radius);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<double>>,
                        rmm::device_uvector<size_t>>
        extract_ego(raft::handle_t const&                               handle,
                    graph_view_t<int32_t, int64_t, false, false> const& graph_view,
                    std::optional<edge_property_view_t<int64_t, double const*>>,
                    int32_t* source_vertex,
                    int32_t  n_subgraphs,
                    int32_t  radius);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>,
                        std::optional<rmm::device_uvector<double>>,
                        rmm::device_uvector<size_t>>
        extract_ego(raft::handle_t const&                               handle,
                    graph_view_t<int64_t, int64_t, false, false> const& graph_view,
                    std::optional<edge_property_view_t<int64_t, double const*>>,
                    int64_t* source_vertex,
                    int64_t  n_subgraphs,
                    int64_t  radius);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<double>>,
                        rmm::device_uvector<size_t>>
        extract_ego(raft::handle_t const&                               handle,
                    graph_view_t<int32_t, int32_t, false, false> const& graph_view,
                    std::optional<edge_property_view_t<int32_t, double const*>>,
                    raft::device_span<int32_t const> source_vertex,
                    int32_t                          radius,
                    bool                             do_expensive_check);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<double>>,
                        rmm::device_uvector<size_t>>
        extract_ego(raft::handle_t const&                               handle,
                    graph_view_t<int32_t, int64_t, false, false> const& graph_view,
                    std::optional<edge_property_view_t<int64_t, double const*>>,
                    raft::device_span<int32_t const> source_vertex,
                    int32_t                          radius,
                    bool                             do_expensive_check);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>,
                        std::optional<rmm::device_uvector<double>>,
                        rmm::device_uvector<size_t>>
        extract_ego(raft::handle_t const&                               handle,
                    graph_view_t<int64_t, int64_t, false, false> const& graph_view,
                    std::optional<edge_property_view_t<int64_t, double const*>>,
                    raft::device_span<int64_t const> source_vertex,
                    int64_t                          radius,
                    bool                             do_expensive_check);

} // namespace rocgraph
