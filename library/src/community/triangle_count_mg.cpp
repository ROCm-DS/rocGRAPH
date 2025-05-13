// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "community/triangle_count_impl.cuh"

namespace rocgraph
{

    template void triangle_count(raft::handle_t const&                              handle,
                                 graph_view_t<int32_t, int32_t, false, true> const& graph_view,
                                 std::optional<raft::device_span<int32_t const>>    vertices,
                                 raft::device_span<int32_t>                         counts,
                                 bool do_expensive_check);

    template void triangle_count(raft::handle_t const&                              handle,
                                 graph_view_t<int32_t, int64_t, false, true> const& graph_view,
                                 std::optional<raft::device_span<int32_t const>>    vertices,
                                 raft::device_span<int64_t>                         counts,
                                 bool do_expensive_check);

    template void triangle_count(raft::handle_t const&                              handle,
                                 graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                                 std::optional<raft::device_span<int64_t const>>    vertices,
                                 raft::device_span<int64_t>                         counts,
                                 bool do_expensive_check);

} // namespace rocgraph
