// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "structure/transpose_graph_storage_impl.cuh"

namespace rocgraph
{

    // MG instantiation

    template std::tuple<
        graph_t<int32_t, int32_t, false, true>,
        std::optional<edge_property_t<graph_view_t<int32_t, int32_t, false, true>, float>>,
        std::optional<rmm::device_uvector<int32_t>>>
        transpose_graph_storage(
            raft::handle_t const&                   handle,
            graph_t<int32_t, int32_t, true, true>&& graph,
            std::optional<edge_property_t<graph_view_t<int32_t, int32_t, true, true>, float>>&&
                                                          edge_weights,
            std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
            bool                                          do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int32_t, true, true>,
        std::optional<edge_property_t<graph_view_t<int32_t, int32_t, true, true>, float>>,
        std::optional<rmm::device_uvector<int32_t>>>
        transpose_graph_storage(
            raft::handle_t const&                    handle,
            graph_t<int32_t, int32_t, false, true>&& graph,
            std::optional<edge_property_t<graph_view_t<int32_t, int32_t, false, true>, float>>&&
                                                          edge_weights,
            std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
            bool                                          do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int64_t, false, true>,
        std::optional<edge_property_t<graph_view_t<int32_t, int64_t, false, true>, float>>,
        std::optional<rmm::device_uvector<int32_t>>>
        transpose_graph_storage(
            raft::handle_t const&                   handle,
            graph_t<int32_t, int64_t, true, true>&& graph,
            std::optional<edge_property_t<graph_view_t<int32_t, int64_t, true, true>, float>>&&
                                                          edge_weights,
            std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
            bool                                          do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int64_t, true, true>,
        std::optional<edge_property_t<graph_view_t<int32_t, int64_t, true, true>, float>>,
        std::optional<rmm::device_uvector<int32_t>>>
        transpose_graph_storage(
            raft::handle_t const&                    handle,
            graph_t<int32_t, int64_t, false, true>&& graph,
            std::optional<edge_property_t<graph_view_t<int32_t, int64_t, false, true>, float>>&&
                                                          edge_weights,
            std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
            bool                                          do_expensive_check);

    template std::tuple<
        graph_t<int64_t, int64_t, false, true>,
        std::optional<edge_property_t<graph_view_t<int64_t, int64_t, false, true>, float>>,
        std::optional<rmm::device_uvector<int64_t>>>
        transpose_graph_storage(
            raft::handle_t const&                   handle,
            graph_t<int64_t, int64_t, true, true>&& graph,
            std::optional<edge_property_t<graph_view_t<int64_t, int64_t, true, true>, float>>&&
                                                          edge_weights,
            std::optional<rmm::device_uvector<int64_t>>&& renumber_map,
            bool                                          do_expensive_check);

    template std::tuple<
        graph_t<int64_t, int64_t, true, true>,
        std::optional<edge_property_t<graph_view_t<int64_t, int64_t, true, true>, float>>,
        std::optional<rmm::device_uvector<int64_t>>>
        transpose_graph_storage(
            raft::handle_t const&                    handle,
            graph_t<int64_t, int64_t, false, true>&& graph,
            std::optional<edge_property_t<graph_view_t<int64_t, int64_t, false, true>, float>>&&
                                                          edge_weights,
            std::optional<rmm::device_uvector<int64_t>>&& renumber_map,
            bool                                          do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int32_t, false, true>,
        std::optional<edge_property_t<graph_view_t<int32_t, int32_t, false, true>, double>>,
        std::optional<rmm::device_uvector<int32_t>>>
        transpose_graph_storage(
            raft::handle_t const&                   handle,
            graph_t<int32_t, int32_t, true, true>&& graph,
            std::optional<edge_property_t<graph_view_t<int32_t, int32_t, true, true>, double>>&&
                                                          edge_weights,
            std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
            bool                                          do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int32_t, true, true>,
        std::optional<edge_property_t<graph_view_t<int32_t, int32_t, true, true>, double>>,
        std::optional<rmm::device_uvector<int32_t>>>
        transpose_graph_storage(
            raft::handle_t const&                    handle,
            graph_t<int32_t, int32_t, false, true>&& graph,
            std::optional<edge_property_t<graph_view_t<int32_t, int32_t, false, true>, double>>&&
                                                          edge_weights,
            std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
            bool                                          do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int64_t, false, true>,
        std::optional<edge_property_t<graph_view_t<int32_t, int64_t, false, true>, double>>,
        std::optional<rmm::device_uvector<int32_t>>>
        transpose_graph_storage(
            raft::handle_t const&                   handle,
            graph_t<int32_t, int64_t, true, true>&& graph,
            std::optional<edge_property_t<graph_view_t<int32_t, int64_t, true, true>, double>>&&
                                                          edge_weights,
            std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
            bool                                          do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int64_t, true, true>,
        std::optional<edge_property_t<graph_view_t<int32_t, int64_t, true, true>, double>>,
        std::optional<rmm::device_uvector<int32_t>>>
        transpose_graph_storage(
            raft::handle_t const&                    handle,
            graph_t<int32_t, int64_t, false, true>&& graph,
            std::optional<edge_property_t<graph_view_t<int32_t, int64_t, false, true>, double>>&&
                                                          edge_weights,
            std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
            bool                                          do_expensive_check);

    template std::tuple<
        graph_t<int64_t, int64_t, false, true>,
        std::optional<edge_property_t<graph_view_t<int64_t, int64_t, false, true>, double>>,
        std::optional<rmm::device_uvector<int64_t>>>
        transpose_graph_storage(
            raft::handle_t const&                   handle,
            graph_t<int64_t, int64_t, true, true>&& graph,
            std::optional<edge_property_t<graph_view_t<int64_t, int64_t, true, true>, double>>&&
                                                          edge_weights,
            std::optional<rmm::device_uvector<int64_t>>&& renumber_map,
            bool                                          do_expensive_check);

    template std::tuple<
        graph_t<int64_t, int64_t, true, true>,
        std::optional<edge_property_t<graph_view_t<int64_t, int64_t, true, true>, double>>,
        std::optional<rmm::device_uvector<int64_t>>>
        transpose_graph_storage(
            raft::handle_t const&                    handle,
            graph_t<int64_t, int64_t, false, true>&& graph,
            std::optional<edge_property_t<graph_view_t<int64_t, int64_t, false, true>, double>>&&
                                                          edge_weights,
            std::optional<rmm::device_uvector<int64_t>>&& renumber_map,
            bool                                          do_expensive_check);

} // namespace rocgraph
