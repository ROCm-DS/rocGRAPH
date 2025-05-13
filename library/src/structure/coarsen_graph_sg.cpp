// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "structure/coarsen_graph_impl.cuh"

namespace rocgraph
{

    // SG instantiation

    template std::tuple<
        graph_t<int32_t, int32_t, true, false>,
        std::optional<edge_property_t<graph_view_t<int32_t, int32_t, true, false>, float>>,
        std::optional<rmm::device_uvector<int32_t>>>
        coarsen_graph(raft::handle_t const&                                      handle,
                      graph_view_t<int32_t, int32_t, true, false> const&         graph_view,
                      std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
                      int32_t const*                                             labels,
                      bool                                                       renumber,
                      bool do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int32_t, false, false>,
        std::optional<edge_property_t<graph_view_t<int32_t, int32_t, false, false>, float>>,
        std::optional<rmm::device_uvector<int32_t>>>
        coarsen_graph(raft::handle_t const&                                      handle,
                      graph_view_t<int32_t, int32_t, false, false> const&        graph_view,
                      std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
                      int32_t const*                                             labels,
                      bool                                                       renumber,
                      bool do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int64_t, true, false>,
        std::optional<edge_property_t<graph_view_t<int32_t, int64_t, true, false>, float>>,
        std::optional<rmm::device_uvector<int32_t>>>
        coarsen_graph(raft::handle_t const&                                      handle,
                      graph_view_t<int32_t, int64_t, true, false> const&         graph_view,
                      std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
                      int32_t const*                                             labels,
                      bool                                                       renumber,
                      bool do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int64_t, false, false>,
        std::optional<edge_property_t<graph_view_t<int32_t, int64_t, false, false>, float>>,
        std::optional<rmm::device_uvector<int32_t>>>
        coarsen_graph(raft::handle_t const&                                      handle,
                      graph_view_t<int32_t, int64_t, false, false> const&        graph_view,
                      std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
                      int32_t const*                                             labels,
                      bool                                                       renumber,
                      bool do_expensive_check);

    template std::tuple<
        graph_t<int64_t, int64_t, true, false>,
        std::optional<edge_property_t<graph_view_t<int64_t, int64_t, true, false>, float>>,
        std::optional<rmm::device_uvector<int64_t>>>
        coarsen_graph(raft::handle_t const&                                      handle,
                      graph_view_t<int64_t, int64_t, true, false> const&         graph_view,
                      std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
                      int64_t const*                                             labels,
                      bool                                                       renumber,
                      bool do_expensive_check);

    template std::tuple<
        graph_t<int64_t, int64_t, false, false>,
        std::optional<edge_property_t<graph_view_t<int64_t, int64_t, false, false>, float>>,
        std::optional<rmm::device_uvector<int64_t>>>
        coarsen_graph(raft::handle_t const&                                      handle,
                      graph_view_t<int64_t, int64_t, false, false> const&        graph_view,
                      std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
                      int64_t const*                                             labels,
                      bool                                                       renumber,
                      bool do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int32_t, true, false>,
        std::optional<edge_property_t<graph_view_t<int32_t, int32_t, true, false>, double>>,
        std::optional<rmm::device_uvector<int32_t>>>
        coarsen_graph(raft::handle_t const&                                       handle,
                      graph_view_t<int32_t, int32_t, true, false> const&          graph_view,
                      std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
                      int32_t const*                                              labels,
                      bool                                                        renumber,
                      bool do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int32_t, false, false>,
        std::optional<edge_property_t<graph_view_t<int32_t, int32_t, false, false>, double>>,
        std::optional<rmm::device_uvector<int32_t>>>
        coarsen_graph(raft::handle_t const&                                       handle,
                      graph_view_t<int32_t, int32_t, false, false> const&         graph_view,
                      std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
                      int32_t const*                                              labels,
                      bool                                                        renumber,
                      bool do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int64_t, true, false>,
        std::optional<edge_property_t<graph_view_t<int32_t, int64_t, true, false>, double>>,
        std::optional<rmm::device_uvector<int32_t>>>
        coarsen_graph(raft::handle_t const&                                       handle,
                      graph_view_t<int32_t, int64_t, true, false> const&          graph_view,
                      std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
                      int32_t const*                                              labels,
                      bool                                                        renumber,
                      bool do_expensive_check);

    template std::tuple<
        graph_t<int32_t, int64_t, false, false>,
        std::optional<edge_property_t<graph_view_t<int32_t, int64_t, false, false>, double>>,
        std::optional<rmm::device_uvector<int32_t>>>
        coarsen_graph(raft::handle_t const&                                       handle,
                      graph_view_t<int32_t, int64_t, false, false> const&         graph_view,
                      std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
                      int32_t const*                                              labels,
                      bool                                                        renumber,
                      bool do_expensive_check);

    template std::tuple<
        graph_t<int64_t, int64_t, true, false>,
        std::optional<edge_property_t<graph_view_t<int64_t, int64_t, true, false>, double>>,
        std::optional<rmm::device_uvector<int64_t>>>
        coarsen_graph(raft::handle_t const&                                       handle,
                      graph_view_t<int64_t, int64_t, true, false> const&          graph_view,
                      std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
                      int64_t const*                                              labels,
                      bool                                                        renumber,
                      bool do_expensive_check);

    template std::tuple<
        graph_t<int64_t, int64_t, false, false>,
        std::optional<edge_property_t<graph_view_t<int64_t, int64_t, false, false>, double>>,
        std::optional<rmm::device_uvector<int64_t>>>
        coarsen_graph(raft::handle_t const&                                       handle,
                      graph_view_t<int64_t, int64_t, false, false> const&         graph_view,
                      std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
                      int64_t const*                                              labels,
                      bool                                                        renumber,
                      bool do_expensive_check);

} // namespace rocgraph
