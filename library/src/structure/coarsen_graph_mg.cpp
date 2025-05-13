// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "structure/coarsen_graph_impl.cuh"

namespace rocgraph
{

    // MG instantiation

    template std::tuple<
        graph_t<int32_t, int32_t, true, true>,
        std::optional<edge_property_t<graph_view_t<int32_t, int32_t, true, true>, float>>,
        std::optional<rmm::device_uvector<int32_t>>>
        coarsen_graph(raft::handle_t const&                                      handle,
                      graph_view_t<int32_t, int32_t, true, true> const&          graph_view,
                      std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
                      int32_t const*                                             labels,
                      bool                                                       renumber,
                      bool do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int32_t, false, true>,
        std::optional<edge_property_t<graph_view_t<int32_t, int32_t, false, true>, float>>,
        std::optional<rmm::device_uvector<int32_t>>>
        coarsen_graph(raft::handle_t const&                                      handle,
                      graph_view_t<int32_t, int32_t, false, true> const&         graph_view,
                      std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
                      int32_t const*                                             labels,
                      bool                                                       renumber,
                      bool do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int64_t, true, true>,
        std::optional<edge_property_t<graph_view_t<int32_t, int64_t, true, true>, float>>,
        std::optional<rmm::device_uvector<int32_t>>>
        coarsen_graph(raft::handle_t const&                                      handle,
                      graph_view_t<int32_t, int64_t, true, true> const&          graph_view,
                      std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
                      int32_t const*                                             labels,
                      bool                                                       renumber,
                      bool do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int64_t, false, true>,
        std::optional<edge_property_t<graph_view_t<int32_t, int64_t, false, true>, float>>,
        std::optional<rmm::device_uvector<int32_t>>>
        coarsen_graph(raft::handle_t const&                                      handle,
                      graph_view_t<int32_t, int64_t, false, true> const&         graph_view,
                      std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
                      int32_t const*                                             labels,
                      bool                                                       renumber,
                      bool do_expensive_check);

    template std::tuple<
        graph_t<int64_t, int64_t, true, true>,
        std::optional<edge_property_t<graph_view_t<int64_t, int64_t, true, true>, float>>,
        std::optional<rmm::device_uvector<int64_t>>>
        coarsen_graph(raft::handle_t const&                                      handle,
                      graph_view_t<int64_t, int64_t, true, true> const&          graph_view,
                      std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
                      int64_t const*                                             labels,
                      bool                                                       renumber,
                      bool do_expensive_check);

    template std::tuple<
        graph_t<int64_t, int64_t, false, true>,
        std::optional<edge_property_t<graph_view_t<int64_t, int64_t, false, true>, float>>,
        std::optional<rmm::device_uvector<int64_t>>>
        coarsen_graph(raft::handle_t const&                                      handle,
                      graph_view_t<int64_t, int64_t, false, true> const&         graph_view,
                      std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
                      int64_t const*                                             labels,
                      bool                                                       renumber,
                      bool do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int32_t, true, true>,
        std::optional<edge_property_t<graph_view_t<int32_t, int32_t, true, true>, double>>,
        std::optional<rmm::device_uvector<int32_t>>>
        coarsen_graph(raft::handle_t const&                                       handle,
                      graph_view_t<int32_t, int32_t, true, true> const&           graph_view,
                      std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
                      int32_t const*                                              labels,
                      bool                                                        renumber,
                      bool do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int32_t, false, true>,
        std::optional<edge_property_t<graph_view_t<int32_t, int32_t, false, true>, double>>,
        std::optional<rmm::device_uvector<int32_t>>>
        coarsen_graph(raft::handle_t const&                                       handle,
                      graph_view_t<int32_t, int32_t, false, true> const&          graph_view,
                      std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
                      int32_t const*                                              labels,
                      bool                                                        renumber,
                      bool do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int64_t, true, true>,
        std::optional<edge_property_t<graph_view_t<int32_t, int64_t, true, true>, double>>,
        std::optional<rmm::device_uvector<int32_t>>>
        coarsen_graph(raft::handle_t const&                                       handle,
                      graph_view_t<int32_t, int64_t, true, true> const&           graph_view,
                      std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
                      int32_t const*                                              labels,
                      bool                                                        renumber,
                      bool do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int64_t, false, true>,
        std::optional<edge_property_t<graph_view_t<int32_t, int64_t, false, true>, double>>,
        std::optional<rmm::device_uvector<int32_t>>>
        coarsen_graph(raft::handle_t const&                                       handle,
                      graph_view_t<int32_t, int64_t, false, true> const&          graph_view,
                      std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
                      int32_t const*                                              labels,
                      bool                                                        renumber,
                      bool do_expensive_check);

    template std::tuple<
        graph_t<int64_t, int64_t, true, true>,
        std::optional<edge_property_t<graph_view_t<int64_t, int64_t, true, true>, double>>,
        std::optional<rmm::device_uvector<int64_t>>>
        coarsen_graph(raft::handle_t const&                                       handle,
                      graph_view_t<int64_t, int64_t, true, true> const&           graph_view,
                      std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
                      int64_t const*                                              labels,
                      bool                                                        renumber,
                      bool do_expensive_check);

    template std::tuple<
        graph_t<int64_t, int64_t, false, true>,
        std::optional<edge_property_t<graph_view_t<int64_t, int64_t, false, true>, double>>,
        std::optional<rmm::device_uvector<int64_t>>>
        coarsen_graph(raft::handle_t const&                                       handle,
                      graph_view_t<int64_t, int64_t, false, true> const&          graph_view,
                      std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
                      int64_t const*                                              labels,
                      bool                                                        renumber,
                      bool do_expensive_check);

} // namespace rocgraph
