// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "traversal/bfs_impl.cuh"

namespace rocgraph
{

    // MG instantiation

    template void bfs(raft::handle_t const&                              handle,
                      graph_view_t<int32_t, int32_t, false, true> const& graph_view,
                      int32_t*                                           distances,
                      int32_t*                                           predecessors,
                      int32_t const*                                     sources,
                      size_t                                             n_sources,
                      bool                                               direction_optimizing,
                      int32_t                                            depth_limit,
                      bool                                               do_expensive_check);

    template void bfs(raft::handle_t const&                              handle,
                      graph_view_t<int32_t, int64_t, false, true> const& graph_view,
                      int32_t*                                           distances,
                      int32_t*                                           predecessors,
                      int32_t const*                                     sources,
                      size_t                                             n_sources,
                      bool                                               direction_optimizing,
                      int32_t                                            depth_limit,
                      bool                                               do_expensive_check);

    template void bfs(raft::handle_t const&                              handle,
                      graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                      int64_t*                                           distances,
                      int64_t*                                           predecessors,
                      int64_t const*                                     sources,
                      size_t                                             n_sources,
                      bool                                               direction_optimizing,
                      int64_t                                            depth_limit,
                      bool                                               do_expensive_check);

} // namespace rocgraph
