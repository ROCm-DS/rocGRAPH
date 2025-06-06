// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "structure/induced_subgraph_impl.cuh"

namespace rocgraph
{

    // MG instantiation

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<float>>,
                        rmm::device_uvector<size_t>>
        extract_induced_subgraphs(
            raft::handle_t const&                                      handle,
            graph_view_t<int32_t, int32_t, false, true> const&         graph_view,
            std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
            raft::device_span<size_t const>                            subgraph_offsets,
            raft::device_span<int32_t const>                           subgraph_vertices,
            bool                                                       do_expensive_check);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<double>>,
                        rmm::device_uvector<size_t>>
        extract_induced_subgraphs(
            raft::handle_t const&                                       handle,
            graph_view_t<int32_t, int32_t, false, true> const&          graph_view,
            std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
            raft::device_span<size_t const>                             subgraph_offsets,
            raft::device_span<int32_t const>                            subgraph_vertices,
            bool                                                        do_expensive_check);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<float>>,
                        rmm::device_uvector<size_t>>
        extract_induced_subgraphs(
            raft::handle_t const&                                      handle,
            graph_view_t<int32_t, int64_t, false, true> const&         graph_view,
            std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
            raft::device_span<size_t const>                            subgraph_offsets,
            raft::device_span<int32_t const>                           subgraph_vertices,
            bool                                                       do_expensive_check);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<double>>,
                        rmm::device_uvector<size_t>>
        extract_induced_subgraphs(
            raft::handle_t const&                                       handle,
            graph_view_t<int32_t, int64_t, false, true> const&          graph_view,
            std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
            raft::device_span<size_t const>                             subgraph_offsets,
            raft::device_span<int32_t const>                            subgraph_vertices,
            bool                                                        do_expensive_check);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>,
                        std::optional<rmm::device_uvector<float>>,
                        rmm::device_uvector<size_t>>
        extract_induced_subgraphs(
            raft::handle_t const&                                      handle,
            graph_view_t<int64_t, int64_t, false, true> const&         graph_view,
            std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
            raft::device_span<size_t const>                            subgraph_offsets,
            raft::device_span<int64_t const>                           subgraph_vertices,
            bool                                                       do_expensive_check);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>,
                        std::optional<rmm::device_uvector<double>>,
                        rmm::device_uvector<size_t>>
        extract_induced_subgraphs(
            raft::handle_t const&                                       handle,
            graph_view_t<int64_t, int64_t, false, true> const&          graph_view,
            std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
            raft::device_span<size_t const>                             subgraph_offsets,
            raft::device_span<int64_t const>                            subgraph_vertices,
            bool                                                        do_expensive_check);

} // namespace rocgraph
