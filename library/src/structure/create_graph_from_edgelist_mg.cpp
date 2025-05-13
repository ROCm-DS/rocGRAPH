// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "structure/create_graph_from_edgelist_impl.cuh"

namespace rocgraph
{

    // explicit instantiations

    template std::tuple<
        rocgraph::graph_t<int32_t, int32_t, false, true>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int32_t, int32_t, false, true>,
                                      float>>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int32_t, int32_t, false, true>,
                                      int32_t>>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int32_t, int32_t, false, true>,
                                      int32_t>>,
        std::optional<rmm::device_uvector<int32_t>>>
        create_graph_from_edgelist<int32_t, int32_t, float, int32_t, int32_t, false, true>(
            raft::handle_t const&                         handle,
            std::optional<rmm::device_uvector<int32_t>>&& vertex_span,
            rmm::device_uvector<int32_t>&&                edgelist_srcs,
            rmm::device_uvector<int32_t>&&                edgelist_dsts,
            std::optional<rmm::device_uvector<float>>&&   edgelist_weights,
            std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_ids,
            std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types,
            graph_properties_t                            graph_properties,
            bool                                          renumber,
            bool                                          do_expensive_check);

    template std::tuple<
        rocgraph::graph_t<int32_t, int32_t, true, true>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int32_t, int32_t, true, true>, float>>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int32_t, int32_t, true, true>,
                                      int32_t>>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int32_t, int32_t, true, true>,
                                      int32_t>>,
        std::optional<rmm::device_uvector<int32_t>>>
        create_graph_from_edgelist<int32_t, int32_t, float, int32_t, int32_t, true, true>(
            raft::handle_t const&                         handle,
            std::optional<rmm::device_uvector<int32_t>>&& vertex_span,
            rmm::device_uvector<int32_t>&&                edgelist_srcs,
            rmm::device_uvector<int32_t>&&                edgelist_dsts,
            std::optional<rmm::device_uvector<float>>&&   edgelist_weights,
            std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_ids,
            std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types,
            graph_properties_t                            graph_properties,
            bool                                          renumber,
            bool                                          do_expensive_check);

    template std::tuple<
        rocgraph::graph_t<int32_t, int32_t, false, true>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int32_t, int32_t, false, true>,
                                      double>>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int32_t, int32_t, false, true>,
                                      int32_t>>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int32_t, int32_t, false, true>,
                                      int32_t>>,
        std::optional<rmm::device_uvector<int32_t>>>
        create_graph_from_edgelist<int32_t, int32_t, double, int32_t, int32_t, false, true>(
            raft::handle_t const&                         handle,
            std::optional<rmm::device_uvector<int32_t>>&& vertex_span,
            rmm::device_uvector<int32_t>&&                edgelist_srcs,
            rmm::device_uvector<int32_t>&&                edgelist_dsts,
            std::optional<rmm::device_uvector<double>>&&  edgelist_weights,
            std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_ids,
            std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types,
            graph_properties_t                            graph_properties,
            bool                                          renumber,
            bool                                          do_expensive_check);

    template std::tuple<
        rocgraph::graph_t<int32_t, int32_t, true, true>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int32_t, int32_t, true, true>,
                                      double>>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int32_t, int32_t, true, true>,
                                      int32_t>>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int32_t, int32_t, true, true>,
                                      int32_t>>,
        std::optional<rmm::device_uvector<int32_t>>>
        create_graph_from_edgelist<int32_t, int32_t, double, int32_t, int32_t, true, true>(
            raft::handle_t const&                         handle,
            std::optional<rmm::device_uvector<int32_t>>&& vertex_span,
            rmm::device_uvector<int32_t>&&                edgelist_srcs,
            rmm::device_uvector<int32_t>&&                edgelist_dsts,
            std::optional<rmm::device_uvector<double>>&&  edgelist_weights,
            std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_ids,
            std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types,
            graph_properties_t                            graph_properties,
            bool                                          renumber,
            bool                                          do_expensive_check);

    template std::tuple<
        rocgraph::graph_t<int32_t, int64_t, false, true>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int32_t, int64_t, false, true>,
                                      float>>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int32_t, int64_t, false, true>,
                                      int64_t>>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int32_t, int64_t, false, true>,
                                      int32_t>>,
        std::optional<rmm::device_uvector<int32_t>>>
        create_graph_from_edgelist<int32_t, int64_t, float, int64_t, int32_t, false, true>(
            raft::handle_t const&                         handle,
            std::optional<rmm::device_uvector<int32_t>>&& vertex_span,
            rmm::device_uvector<int32_t>&&                edgelist_srcs,
            rmm::device_uvector<int32_t>&&                edgelist_dsts,
            std::optional<rmm::device_uvector<float>>&&   edgelist_weights,
            std::optional<rmm::device_uvector<int64_t>>&& edgelist_edge_ids,
            std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types,
            graph_properties_t                            graph_properties,
            bool                                          renumber,
            bool                                          do_expensive_check);

    template std::tuple<
        rocgraph::graph_t<int32_t, int64_t, true, true>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int32_t, int64_t, true, true>, float>>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int32_t, int64_t, true, true>,
                                      int64_t>>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int32_t, int64_t, true, true>,
                                      int32_t>>,
        std::optional<rmm::device_uvector<int32_t>>>
        create_graph_from_edgelist<int32_t, int64_t, float, int64_t, int32_t, true, true>(
            raft::handle_t const&                         handle,
            std::optional<rmm::device_uvector<int32_t>>&& vertex_span,
            rmm::device_uvector<int32_t>&&                edgelist_srcs,
            rmm::device_uvector<int32_t>&&                edgelist_dsts,
            std::optional<rmm::device_uvector<float>>&&   edgelist_weights,
            std::optional<rmm::device_uvector<int64_t>>&& edgelist_edge_ids,
            std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types,
            graph_properties_t                            graph_properties,
            bool                                          renumber,
            bool                                          do_expensive_check);

    template std::tuple<
        rocgraph::graph_t<int32_t, int64_t, false, true>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int32_t, int64_t, false, true>,
                                      double>>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int32_t, int64_t, false, true>,
                                      int64_t>>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int32_t, int64_t, false, true>,
                                      int32_t>>,
        std::optional<rmm::device_uvector<int32_t>>>
        create_graph_from_edgelist<int32_t, int64_t, double, int64_t, int32_t, false, true>(
            raft::handle_t const&                         handle,
            std::optional<rmm::device_uvector<int32_t>>&& vertex_span,
            rmm::device_uvector<int32_t>&&                edgelist_srcs,
            rmm::device_uvector<int32_t>&&                edgelist_dsts,
            std::optional<rmm::device_uvector<double>>&&  edgelist_weights,
            std::optional<rmm::device_uvector<int64_t>>&& edgelist_edge_ids,
            std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types,
            graph_properties_t                            graph_properties,
            bool                                          renumber,
            bool                                          do_expensive_check);

    template std::tuple<
        rocgraph::graph_t<int32_t, int64_t, true, true>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int32_t, int64_t, true, true>,
                                      double>>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int32_t, int64_t, true, true>,
                                      int64_t>>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int32_t, int64_t, true, true>,
                                      int32_t>>,
        std::optional<rmm::device_uvector<int32_t>>>
        create_graph_from_edgelist<int32_t, int64_t, double, int64_t, int32_t, true, true>(
            raft::handle_t const&                         handle,
            std::optional<rmm::device_uvector<int32_t>>&& vertex_span,
            rmm::device_uvector<int32_t>&&                edgelist_srcs,
            rmm::device_uvector<int32_t>&&                edgelist_dsts,
            std::optional<rmm::device_uvector<double>>&&  edgelist_weights,
            std::optional<rmm::device_uvector<int64_t>>&& edgelist_edge_ids,
            std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types,
            graph_properties_t                            graph_properties,
            bool                                          renumber,
            bool                                          do_expensive_check);

    template std::tuple<
        rocgraph::graph_t<int64_t, int64_t, false, true>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int64_t, int64_t, false, true>,
                                      float>>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int64_t, int64_t, false, true>,
                                      int64_t>>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int64_t, int64_t, false, true>,
                                      int32_t>>,
        std::optional<rmm::device_uvector<int64_t>>>
        create_graph_from_edgelist<int64_t, int64_t, float, int64_t, int32_t, false, true>(
            raft::handle_t const&                         handle,
            std::optional<rmm::device_uvector<int64_t>>&& vertex_span,
            rmm::device_uvector<int64_t>&&                edgelist_srcs,
            rmm::device_uvector<int64_t>&&                edgelist_dsts,
            std::optional<rmm::device_uvector<float>>&&   edgelist_weights,
            std::optional<rmm::device_uvector<int64_t>>&& edgelist_edge_ids,
            std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types,
            graph_properties_t                            graph_properties,
            bool                                          renumber,
            bool                                          do_expensive_check);

    template std::tuple<
        rocgraph::graph_t<int64_t, int64_t, true, true>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int64_t, int64_t, true, true>, float>>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int64_t, int64_t, true, true>,
                                      int64_t>>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int64_t, int64_t, true, true>,
                                      int32_t>>,
        std::optional<rmm::device_uvector<int64_t>>>
        create_graph_from_edgelist<int64_t, int64_t, float, int64_t, int32_t, true, true>(
            raft::handle_t const&                         handle,
            std::optional<rmm::device_uvector<int64_t>>&& vertex_span,
            rmm::device_uvector<int64_t>&&                edgelist_srcs,
            rmm::device_uvector<int64_t>&&                edgelist_dsts,
            std::optional<rmm::device_uvector<float>>&&   edgelist_weights,
            std::optional<rmm::device_uvector<int64_t>>&& edgelist_edge_ids,
            std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types,
            graph_properties_t                            graph_properties,
            bool                                          renumber,
            bool                                          do_expensive_check);

    template std::tuple<
        rocgraph::graph_t<int64_t, int64_t, false, true>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int64_t, int64_t, false, true>,
                                      double>>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int64_t, int64_t, false, true>,
                                      int64_t>>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int64_t, int64_t, false, true>,
                                      int32_t>>,
        std::optional<rmm::device_uvector<int64_t>>>
        create_graph_from_edgelist<int64_t, int64_t, double, int64_t, int32_t, false, true>(
            raft::handle_t const&                         handle,
            std::optional<rmm::device_uvector<int64_t>>&& vertex_span,
            rmm::device_uvector<int64_t>&&                edgelist_srcs,
            rmm::device_uvector<int64_t>&&                edgelist_dsts,
            std::optional<rmm::device_uvector<double>>&&  edgelist_weights,
            std::optional<rmm::device_uvector<int64_t>>&& edgelist_edge_ids,
            std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types,
            graph_properties_t                            graph_properties,
            bool                                          renumber,
            bool                                          do_expensive_check);

    template std::tuple<
        rocgraph::graph_t<int64_t, int64_t, true, true>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int64_t, int64_t, true, true>,
                                      double>>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int64_t, int64_t, true, true>,
                                      int64_t>>,
        std::optional<
            rocgraph::edge_property_t<rocgraph::graph_view_t<int64_t, int64_t, true, true>,
                                      int32_t>>,
        std::optional<rmm::device_uvector<int64_t>>>
        create_graph_from_edgelist<int64_t, int64_t, double, int64_t, int32_t, true, true>(
            raft::handle_t const&                         handle,
            std::optional<rmm::device_uvector<int64_t>>&& vertex_span,
            rmm::device_uvector<int64_t>&&                edgelist_srcs,
            rmm::device_uvector<int64_t>&&                edgelist_dsts,
            std::optional<rmm::device_uvector<double>>&&  edgelist_weights,
            std::optional<rmm::device_uvector<int64_t>>&& edgelist_edge_ids,
            std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types,
            graph_properties_t                            graph_properties,
            bool                                          renumber,
            bool                                          do_expensive_check);

} // namespace rocgraph
