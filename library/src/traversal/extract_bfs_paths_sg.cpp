// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "traversal/extract_bfs_paths_impl.cuh"

namespace rocgraph
{

    // SG instantiation

    template std::tuple<rmm::device_uvector<int32_t>, int32_t>
        extract_bfs_paths(raft::handle_t const&                               handle,
                          graph_view_t<int32_t, int32_t, false, false> const& graph_view,
                          int32_t const*                                      distances,
                          int32_t const*                                      predecessors,
                          int32_t const*                                      destinations,
                          size_t                                              n_destinations);

    template std::tuple<rmm::device_uvector<int32_t>, int32_t>
        extract_bfs_paths(raft::handle_t const&                               handle,
                          graph_view_t<int32_t, int64_t, false, false> const& graph_view,
                          int32_t const*                                      distances,
                          int32_t const*                                      predecessors,
                          int32_t const*                                      destinations,
                          size_t                                              n_destinations);

    template std::tuple<rmm::device_uvector<int64_t>, int64_t>
        extract_bfs_paths(raft::handle_t const&                               handle,
                          graph_view_t<int64_t, int64_t, false, false> const& graph_view,
                          int64_t const*                                      distances,
                          int64_t const*                                      predecessors,
                          int64_t const*                                      destinations,
                          size_t                                              n_destinations);

} // namespace rocgraph
