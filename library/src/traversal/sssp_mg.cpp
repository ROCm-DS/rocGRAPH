// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "traversal/sssp_impl.cuh"

namespace rocgraph
{

    // MG instantiation

    template void sssp(raft::handle_t const&                              handle,
                       graph_view_t<int32_t, int32_t, false, true> const& graph_view,
                       edge_property_view_t<int32_t, float const*>        edge_weight_view,
                       float*                                             distances,
                       int32_t*                                           predecessors,
                       int32_t                                            source_vertex,
                       float                                              cutoff,
                       bool                                               do_expensive_check);

    template void sssp(raft::handle_t const&                              handle,
                       graph_view_t<int32_t, int32_t, false, true> const& graph_view,
                       edge_property_view_t<int32_t, double const*>       edge_weight_view,
                       double*                                            distances,
                       int32_t*                                           predecessors,
                       int32_t                                            source_vertex,
                       double                                             cutoff,
                       bool                                               do_expensive_check);

    template void sssp(raft::handle_t const&                              handle,
                       graph_view_t<int32_t, int64_t, false, true> const& graph_view,
                       edge_property_view_t<int64_t, float const*>        edge_weight_view,
                       float*                                             distances,
                       int32_t*                                           predecessors,
                       int32_t                                            source_vertex,
                       float                                              cutoff,
                       bool                                               do_expensive_check);

    template void sssp(raft::handle_t const&                              handle,
                       graph_view_t<int32_t, int64_t, false, true> const& graph_view,
                       edge_property_view_t<int64_t, double const*>       edge_weight_view,
                       double*                                            distances,
                       int32_t*                                           predecessors,
                       int32_t                                            source_vertex,
                       double                                             cutoff,
                       bool                                               do_expensive_check);

    template void sssp(raft::handle_t const&                              handle,
                       graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                       edge_property_view_t<int64_t, float const*>        edge_weight_view,
                       float*                                             distances,
                       int64_t*                                           predecessors,
                       int64_t                                            source_vertex,
                       float                                              cutoff,
                       bool                                               do_expensive_check);

    template void sssp(raft::handle_t const&                              handle,
                       graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                       edge_property_view_t<int64_t, double const*>       edge_weight_view,
                       double*                                            distances,
                       int64_t*                                           predecessors,
                       int64_t                                            source_vertex,
                       double                                             cutoff,
                       bool                                               do_expensive_check);

} // namespace rocgraph
