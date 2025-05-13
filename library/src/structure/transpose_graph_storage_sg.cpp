// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "structure/transpose_graph_storage_impl.cuh"

namespace rocgraph
{

    // SG instantiation

    template std::tuple<
        graph_t<int32_t, int32_t, false, false>,
        std::optional<edge_property_t<graph_view_t<int32_t, int32_t, false, false>, float>>,
        std::optional<rmm::device_uvector<int32_t>>>
        transpose_graph_storage(
            raft::handle_t const&                    handle,
            graph_t<int32_t, int32_t, true, false>&& graph,
            std::optional<edge_property_t<graph_view_t<int32_t, int32_t, true, false>, float>>&&
                                                          edge_weights,
            std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
            bool                                          do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int32_t, true, false>,
        std::optional<edge_property_t<graph_view_t<int32_t, int32_t, true, false>, float>>,
        std::optional<rmm::device_uvector<int32_t>>>
        transpose_graph_storage(
            raft::handle_t const&                     handle,
            graph_t<int32_t, int32_t, false, false>&& graph,
            std::optional<edge_property_t<graph_view_t<int32_t, int32_t, false, false>, float>>&&
                                                          edge_weights,
            std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
            bool                                          do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int64_t, false, false>,
        std::optional<edge_property_t<graph_view_t<int32_t, int64_t, false, false>, float>>,
        std::optional<rmm::device_uvector<int32_t>>>
        transpose_graph_storage(
            raft::handle_t const&                    handle,
            graph_t<int32_t, int64_t, true, false>&& graph,
            std::optional<edge_property_t<graph_view_t<int32_t, int64_t, true, false>, float>>&&
                                                          edge_weights,
            std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
            bool                                          do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int64_t, true, false>,
        std::optional<edge_property_t<graph_view_t<int32_t, int64_t, true, false>, float>>,
        std::optional<rmm::device_uvector<int32_t>>>
        transpose_graph_storage(
            raft::handle_t const&                     handle,
            graph_t<int32_t, int64_t, false, false>&& graph,
            std::optional<edge_property_t<graph_view_t<int32_t, int64_t, false, false>, float>>&&
                                                          edge_weights,
            std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
            bool                                          do_expensive_check);

    template std::tuple<
        graph_t<int64_t, int64_t, false, false>,
        std::optional<edge_property_t<graph_view_t<int64_t, int64_t, false, false>, float>>,
        std::optional<rmm::device_uvector<int64_t>>>
        transpose_graph_storage(
            raft::handle_t const&                    handle,
            graph_t<int64_t, int64_t, true, false>&& graph,
            std::optional<edge_property_t<graph_view_t<int64_t, int64_t, true, false>, float>>&&
                                                          edge_weights,
            std::optional<rmm::device_uvector<int64_t>>&& renumber_map,
            bool                                          do_expensive_check);

    template std::tuple<
        graph_t<int64_t, int64_t, true, false>,
        std::optional<edge_property_t<graph_view_t<int64_t, int64_t, true, false>, float>>,
        std::optional<rmm::device_uvector<int64_t>>>
        transpose_graph_storage(
            raft::handle_t const&                     handle,
            graph_t<int64_t, int64_t, false, false>&& graph,
            std::optional<edge_property_t<graph_view_t<int64_t, int64_t, false, false>, float>>&&
                                                          edge_weights,
            std::optional<rmm::device_uvector<int64_t>>&& renumber_map,
            bool                                          do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int32_t, false, false>,
        std::optional<edge_property_t<graph_view_t<int32_t, int32_t, false, false>, double>>,
        std::optional<rmm::device_uvector<int32_t>>>
        transpose_graph_storage(
            raft::handle_t const&                    handle,
            graph_t<int32_t, int32_t, true, false>&& graph,
            std::optional<edge_property_t<graph_view_t<int32_t, int32_t, true, false>, double>>&&
                                                          edge_weights,
            std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
            bool                                          do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int32_t, true, false>,
        std::optional<edge_property_t<graph_view_t<int32_t, int32_t, true, false>, double>>,
        std::optional<rmm::device_uvector<int32_t>>>
        transpose_graph_storage(
            raft::handle_t const&                     handle,
            graph_t<int32_t, int32_t, false, false>&& graph,
            std::optional<edge_property_t<graph_view_t<int32_t, int32_t, false, false>, double>>&&
                                                          edge_weights,
            std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
            bool                                          do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int64_t, false, false>,
        std::optional<edge_property_t<graph_view_t<int32_t, int64_t, false, false>, double>>,
        std::optional<rmm::device_uvector<int32_t>>>
        transpose_graph_storage(
            raft::handle_t const&                    handle,
            graph_t<int32_t, int64_t, true, false>&& graph,
            std::optional<edge_property_t<graph_view_t<int32_t, int64_t, true, false>, double>>&&
                                                          edge_weights,
            std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
            bool                                          do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int64_t, true, false>,
        std::optional<edge_property_t<graph_view_t<int32_t, int64_t, true, false>, double>>,
        std::optional<rmm::device_uvector<int32_t>>>
        transpose_graph_storage(
            raft::handle_t const&                     handle,
            graph_t<int32_t, int64_t, false, false>&& graph,
            std::optional<edge_property_t<graph_view_t<int32_t, int64_t, false, false>, double>>&&
                                                          edge_weights,
            std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
            bool                                          do_expensive_check);

    template std::tuple<
        graph_t<int64_t, int64_t, false, false>,
        std::optional<edge_property_t<graph_view_t<int64_t, int64_t, false, false>, double>>,
        std::optional<rmm::device_uvector<int64_t>>>
        transpose_graph_storage(
            raft::handle_t const&                    handle,
            graph_t<int64_t, int64_t, true, false>&& graph,
            std::optional<edge_property_t<graph_view_t<int64_t, int64_t, true, false>, double>>&&
                                                          edge_weights,
            std::optional<rmm::device_uvector<int64_t>>&& renumber_map,
            bool                                          do_expensive_check);

    template std::tuple<
        graph_t<int64_t, int64_t, true, false>,
        std::optional<edge_property_t<graph_view_t<int64_t, int64_t, true, false>, double>>,
        std::optional<rmm::device_uvector<int64_t>>>
        transpose_graph_storage(
            raft::handle_t const&                     handle,
            graph_t<int64_t, int64_t, false, false>&& graph,
            std::optional<edge_property_t<graph_view_t<int64_t, int64_t, false, false>, double>>&&
                                                          edge_weights,
            std::optional<rmm::device_uvector<int64_t>>&& renumber_map,
            bool                                          do_expensive_check);

} // namespace rocgraph
