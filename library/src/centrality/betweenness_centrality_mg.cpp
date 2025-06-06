// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "centrality/betweenness_centrality_impl.cuh"

namespace rocgraph
{

    // MG instantiation

    template rmm::device_uvector<float> betweenness_centrality(
        raft::handle_t const&                                      handle,
        graph_view_t<int32_t, int32_t, false, true> const&         graph_view,
        std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
        std::optional<raft::device_span<int32_t const>>            vertices,
        bool const                                                 normalized,
        bool const                                                 include_endpoints,
        bool                                                       do_expensive_check);

    template rmm::device_uvector<float> betweenness_centrality(
        raft::handle_t const&                                      handle,
        graph_view_t<int32_t, int64_t, false, true> const&         graph_view,
        std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
        std::optional<raft::device_span<int32_t const>>            vertices,
        bool const                                                 normalized,
        bool const                                                 include_endpoints,
        bool                                                       do_expensive_check);

    template rmm::device_uvector<float> betweenness_centrality(
        raft::handle_t const&                                      handle,
        graph_view_t<int64_t, int64_t, false, true> const&         graph_view,
        std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
        std::optional<raft::device_span<int64_t const>>            vertices,
        bool const                                                 normalized,
        bool const                                                 include_endpoints,
        bool                                                       do_expensive_check);

    template rmm::device_uvector<double> betweenness_centrality(
        raft::handle_t const&                                       handle,
        graph_view_t<int32_t, int32_t, false, true> const&          graph_view,
        std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
        std::optional<raft::device_span<int32_t const>>             vertices,
        bool const                                                  normalized,
        bool const                                                  include_endpoints,
        bool                                                        do_expensive_check);

    template rmm::device_uvector<double> betweenness_centrality(
        raft::handle_t const&                                       handle,
        graph_view_t<int32_t, int64_t, false, true> const&          graph_view,
        std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
        std::optional<raft::device_span<int32_t const>>             vertices,
        bool const                                                  normalized,
        bool const                                                  include_endpoints,
        bool                                                        do_expensive_check);

    template rmm::device_uvector<double> betweenness_centrality(
        raft::handle_t const&                                       handle,
        graph_view_t<int64_t, int64_t, false, true> const&          graph_view,
        std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
        std::optional<raft::device_span<int64_t const>>             vertices,
        bool const                                                  normalized,
        bool const                                                  include_endpoints,
        bool                                                        do_expensive_check);

    template edge_property_t<graph_view_t<int32_t, int32_t, false, true>, float>
        edge_betweenness_centrality(
            const raft::handle_t&                                      handle,
            graph_view_t<int32_t, int32_t, false, true> const&         graph_view,
            std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
            std::optional<raft::device_span<int32_t const>>            vertices,
            bool const                                                 normalized,
            bool const                                                 do_expensive_check);

    template edge_property_t<graph_view_t<int32_t, int64_t, false, true>, float>
        edge_betweenness_centrality(
            const raft::handle_t&                                      handle,
            graph_view_t<int32_t, int64_t, false, true> const&         graph_view,
            std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
            std::optional<raft::device_span<int32_t const>>            vertices,
            bool const                                                 normalized,
            bool const                                                 do_expensive_check);

    template edge_property_t<graph_view_t<int64_t, int64_t, false, true>, float>
        edge_betweenness_centrality(
            const raft::handle_t&                                      handle,
            graph_view_t<int64_t, int64_t, false, true> const&         graph_view,
            std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
            std::optional<raft::device_span<int64_t const>>            vertices,
            bool const                                                 normalized,
            bool const                                                 do_expensive_check);

    template edge_property_t<graph_view_t<int32_t, int32_t, false, true>, double>
        edge_betweenness_centrality(
            const raft::handle_t&                                       handle,
            graph_view_t<int32_t, int32_t, false, true> const&          graph_view,
            std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
            std::optional<raft::device_span<int32_t const>>             vertices,
            bool const                                                  normalized,
            bool const                                                  do_expensive_check);

    template edge_property_t<graph_view_t<int32_t, int64_t, false, true>, double>
        edge_betweenness_centrality(
            const raft::handle_t&                                       handle,
            graph_view_t<int32_t, int64_t, false, true> const&          graph_view,
            std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
            std::optional<raft::device_span<int32_t const>>             vertices,
            bool const                                                  normalized,
            bool const                                                  do_expensive_check);

    template edge_property_t<graph_view_t<int64_t, int64_t, false, true>, double>
        edge_betweenness_centrality(
            const raft::handle_t&                                       handle,
            graph_view_t<int64_t, int64_t, false, true> const&          graph_view,
            std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
            std::optional<raft::device_span<int64_t const>>             vertices,
            bool const                                                  normalized,
            bool const                                                  do_expensive_check);

} // namespace rocgraph
