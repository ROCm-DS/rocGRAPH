// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_graph_helper_impl.cuh"

namespace rocgraph
{
    namespace c_api
    {

        template rmm::device_uvector<int32_t>
            expand_sparse_offsets(raft::device_span<int32_t const> offsets,
                                  int32_t                          base_vertex_id,
                                  rmm::cuda_stream_view const&     stream);

        template rmm::device_uvector<int32_t>
            expand_sparse_offsets(raft::device_span<int64_t const> offsets,
                                  int32_t                          base_vertex_id,
                                  rmm::cuda_stream_view const&     stream);

        template rmm::device_uvector<int64_t>
            expand_sparse_offsets(raft::device_span<int64_t const> offsets,
                                  int64_t                          base_vertex_id,
                                  rmm::cuda_stream_view const&     stream);

        template rmm::device_uvector<int32_t>
            expand_sparse_offsets(raft::device_span<size_t const> offsets,
                                  int32_t                         base_vertex_id,
                                  rmm::cuda_stream_view const&    stream);

        template rmm::device_uvector<int64_t>
            expand_sparse_offsets(raft::device_span<size_t const> offsets,
                                  int64_t                         base_vertex_id,
                                  rmm::cuda_stream_view const&    stream);

        template edge_property_t<rocgraph::graph_view_t<int32_t, int32_t, false, false>, float>
            create_constant_edge_property(
                raft::handle_t const&                                         handle,
                rocgraph::graph_view_t<int32_t, int32_t, false, false> const& graph_view,
                float                                                         constant_value);

        template edge_property_t<rocgraph::graph_view_t<int32_t, int64_t, false, false>, float>
            create_constant_edge_property(
                raft::handle_t const&                                         handle,
                rocgraph::graph_view_t<int32_t, int64_t, false, false> const& graph_view,
                float                                                         constant_value);

        template edge_property_t<rocgraph::graph_view_t<int64_t, int64_t, false, false>, float>
            create_constant_edge_property(
                raft::handle_t const&                                         handle,
                rocgraph::graph_view_t<int64_t, int64_t, false, false> const& graph_view,
                float                                                         constant_value);

        template edge_property_t<rocgraph::graph_view_t<int32_t, int32_t, true, false>, float>
            create_constant_edge_property(
                raft::handle_t const&                                        handle,
                rocgraph::graph_view_t<int32_t, int32_t, true, false> const& graph_view,
                float                                                        constant_value);

        template edge_property_t<rocgraph::graph_view_t<int32_t, int64_t, true, false>, float>
            create_constant_edge_property(
                raft::handle_t const&                                        handle,
                rocgraph::graph_view_t<int32_t, int64_t, true, false> const& graph_view,
                float                                                        constant_value);

        template edge_property_t<rocgraph::graph_view_t<int64_t, int64_t, true, false>, float>
            create_constant_edge_property(
                raft::handle_t const&                                        handle,
                rocgraph::graph_view_t<int64_t, int64_t, true, false> const& graph_view,
                float                                                        constant_value);

        template edge_property_t<rocgraph::graph_view_t<int32_t, int32_t, false, false>, double>
            create_constant_edge_property(
                raft::handle_t const&                                         handle,
                rocgraph::graph_view_t<int32_t, int32_t, false, false> const& graph_view,
                double                                                        constant_value);

        template edge_property_t<rocgraph::graph_view_t<int32_t, int64_t, false, false>, double>
            create_constant_edge_property(
                raft::handle_t const&                                         handle,
                rocgraph::graph_view_t<int32_t, int64_t, false, false> const& graph_view,
                double                                                        constant_value);

        template edge_property_t<rocgraph::graph_view_t<int64_t, int64_t, false, false>, double>
            create_constant_edge_property(
                raft::handle_t const&                                         handle,
                rocgraph::graph_view_t<int64_t, int64_t, false, false> const& graph_view,
                double                                                        constant_value);

        template edge_property_t<rocgraph::graph_view_t<int32_t, int32_t, true, false>, double>
            create_constant_edge_property(
                raft::handle_t const&                                        handle,
                rocgraph::graph_view_t<int32_t, int32_t, true, false> const& graph_view,
                double                                                       constant_value);

        template edge_property_t<rocgraph::graph_view_t<int32_t, int64_t, true, false>, double>
            create_constant_edge_property(
                raft::handle_t const&                                        handle,
                rocgraph::graph_view_t<int32_t, int64_t, true, false> const& graph_view,
                double                                                       constant_value);

        template edge_property_t<rocgraph::graph_view_t<int64_t, int64_t, true, false>, double>
            create_constant_edge_property(
                raft::handle_t const&                                        handle,
                rocgraph::graph_view_t<int64_t, int64_t, true, false> const& graph_view,
                double                                                       constant_value);

    } // namespace c_api
} // namespace rocgraph
