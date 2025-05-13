// Copyright (C) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_graph_helper_impl.cuh"

namespace rocgraph
{
    namespace c_api
    {

        template edge_property_t<rocgraph::graph_view_t<int32_t, int32_t, false, true>, float>
            create_constant_edge_property(
                raft::handle_t const&                                        handle,
                rocgraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
                float                                                        constant_value);

        template edge_property_t<rocgraph::graph_view_t<int32_t, int64_t, false, true>, float>
            create_constant_edge_property(
                raft::handle_t const&                                        handle,
                rocgraph::graph_view_t<int32_t, int64_t, false, true> const& graph_view,
                float                                                        constant_value);

        template edge_property_t<rocgraph::graph_view_t<int64_t, int64_t, false, true>, float>
            create_constant_edge_property(
                raft::handle_t const&                                        handle,
                rocgraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                float                                                        constant_value);

        template edge_property_t<rocgraph::graph_view_t<int32_t, int32_t, true, true>, float>
            create_constant_edge_property(
                raft::handle_t const&                                       handle,
                rocgraph::graph_view_t<int32_t, int32_t, true, true> const& graph_view,
                float                                                       constant_value);

        template edge_property_t<rocgraph::graph_view_t<int32_t, int64_t, true, true>, float>
            create_constant_edge_property(
                raft::handle_t const&                                       handle,
                rocgraph::graph_view_t<int32_t, int64_t, true, true> const& graph_view,
                float                                                       constant_value);

        template edge_property_t<rocgraph::graph_view_t<int64_t, int64_t, true, true>, float>
            create_constant_edge_property(
                raft::handle_t const&                                       handle,
                rocgraph::graph_view_t<int64_t, int64_t, true, true> const& graph_view,
                float                                                       constant_value);

        template edge_property_t<rocgraph::graph_view_t<int32_t, int32_t, false, true>, double>
            create_constant_edge_property(
                raft::handle_t const&                                        handle,
                rocgraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
                double                                                       constant_value);

        template edge_property_t<rocgraph::graph_view_t<int32_t, int64_t, false, true>, double>
            create_constant_edge_property(
                raft::handle_t const&                                        handle,
                rocgraph::graph_view_t<int32_t, int64_t, false, true> const& graph_view,
                double                                                       constant_value);

        template edge_property_t<rocgraph::graph_view_t<int64_t, int64_t, false, true>, double>
            create_constant_edge_property(
                raft::handle_t const&                                        handle,
                rocgraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                double                                                       constant_value);

        template edge_property_t<rocgraph::graph_view_t<int32_t, int32_t, true, true>, double>
            create_constant_edge_property(
                raft::handle_t const&                                       handle,
                rocgraph::graph_view_t<int32_t, int32_t, true, true> const& graph_view,
                double                                                      constant_value);

        template edge_property_t<rocgraph::graph_view_t<int32_t, int64_t, true, true>, double>
            create_constant_edge_property(
                raft::handle_t const&                                       handle,
                rocgraph::graph_view_t<int32_t, int64_t, true, true> const& graph_view,
                double                                                      constant_value);

        template edge_property_t<rocgraph::graph_view_t<int64_t, int64_t, true, true>, double>
            create_constant_edge_property(
                raft::handle_t const&                                       handle,
                rocgraph::graph_view_t<int64_t, int64_t, true, true> const& graph_view,
                double                                                      constant_value);

    } // namespace c_api
} // namespace rocgraph
