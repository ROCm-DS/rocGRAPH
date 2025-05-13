// Copyright (C) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "structure/select_random_vertices_impl.hpp"

namespace rocgraph
{

    template rmm::device_uvector<int32_t>
        select_random_vertices(raft::handle_t const&                              handle,
                               graph_view_t<int32_t, int32_t, false, true> const& graph_view,
                               std::optional<raft::device_span<int32_t const>>    given_set,
                               raft::random::RngState&                            rng_state,
                               size_t                                             select_count,
                               bool                                               with_replacement,
                               bool                                               sort_vertices,
                               bool do_expensive_check);

    template rmm::device_uvector<int32_t>
        select_random_vertices(raft::handle_t const&                              handle,
                               graph_view_t<int32_t, int64_t, false, true> const& graph_view,
                               std::optional<raft::device_span<int32_t const>>    given_set,
                               raft::random::RngState&                            rng_state,
                               size_t                                             select_count,
                               bool                                               with_replacement,
                               bool                                               sort_vertices,
                               bool do_expensive_check);

    template rmm::device_uvector<int64_t>
        select_random_vertices(raft::handle_t const&                              handle,
                               graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                               std::optional<raft::device_span<int64_t const>>    given_set,
                               raft::random::RngState&                            rng_state,
                               size_t                                             select_count,
                               bool                                               with_replacement,
                               bool                                               sort_vertices,
                               bool do_expensive_check);

    template rmm::device_uvector<int32_t>
        select_random_vertices(raft::handle_t const&                             handle,
                               graph_view_t<int32_t, int32_t, true, true> const& graph_view,
                               std::optional<raft::device_span<int32_t const>>   given_set,
                               raft::random::RngState&                           rng_state,
                               size_t                                            select_count,
                               bool                                              with_replacement,
                               bool                                              sort_vertices,
                               bool do_expensive_check);

    template rmm::device_uvector<int32_t>
        select_random_vertices(raft::handle_t const&                             handle,
                               graph_view_t<int32_t, int64_t, true, true> const& graph_view,
                               std::optional<raft::device_span<int32_t const>>   given_set,
                               raft::random::RngState&                           rng_state,
                               size_t                                            select_count,
                               bool                                              with_replacement,
                               bool                                              sort_vertices,
                               bool do_expensive_check);

    template rmm::device_uvector<int64_t>
        select_random_vertices(raft::handle_t const&                             handle,
                               graph_view_t<int64_t, int64_t, true, true> const& graph_view,
                               std::optional<raft::device_span<int64_t const>>   given_set,
                               raft::random::RngState&                           rng_state,
                               size_t                                            select_count,
                               bool                                              with_replacement,
                               bool                                              sort_vertices,
                               bool do_expensive_check);

} // namespace rocgraph
