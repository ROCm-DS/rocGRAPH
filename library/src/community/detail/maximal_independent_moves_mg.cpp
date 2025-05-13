// Copyright (C) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "maximal_independent_moves.cuh"

namespace rocgraph
{
    namespace detail
    {

        template rmm::device_uvector<int32_t> maximal_independent_moves(
            raft::handle_t const&                              handle,
            graph_view_t<int32_t, int32_t, false, true> const& decision_graph_view,
            raft::random::RngState&                            rng_state);

        template rmm::device_uvector<int32_t> maximal_independent_moves(
            raft::handle_t const&                              handle,
            graph_view_t<int32_t, int64_t, false, true> const& decision_graph_view,
            raft::random::RngState&                            rng_state);

        template rmm::device_uvector<int64_t> maximal_independent_moves(
            raft::handle_t const&                              handle,
            graph_view_t<int64_t, int64_t, false, true> const& decision_graph_view,
            raft::random::RngState&                            rng_state);

    } // namespace detail

} // namespace rocgraph
