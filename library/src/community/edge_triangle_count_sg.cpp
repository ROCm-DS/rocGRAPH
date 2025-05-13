// Copyright (C) 2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "community/edge_triangle_count_impl.cuh"

namespace rocgraph
{

    // SG instantiation
    template rmm::device_uvector<int32_t> edge_triangle_count(
        raft::handle_t const&                                         handle,
        rocgraph::graph_view_t<int32_t, int32_t, false, false> const& graph_view,
        raft::device_span<int32_t>                                    edgelist_srcs,
        raft::device_span<int32_t>                                    edgelist_dsts);

    template rmm::device_uvector<int64_t> edge_triangle_count(
        raft::handle_t const&                                         handle,
        rocgraph::graph_view_t<int32_t, int64_t, false, false> const& graph_view,
        raft::device_span<int32_t>                                    edgelist_srcs,
        raft::device_span<int32_t>                                    edgelist_dsts);

    template rmm::device_uvector<int64_t> edge_triangle_count(
        raft::handle_t const&                                         handle,
        rocgraph::graph_view_t<int64_t, int64_t, false, false> const& graph_view,
        raft::device_span<int64_t>                                    edgelist_srcs,
        raft::device_span<int64_t>                                    edgelist_dsts);

} // namespace rocgraph
