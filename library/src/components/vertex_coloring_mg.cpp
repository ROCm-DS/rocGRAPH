// Copyright (C) 2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "vertex_coloring_impl.cuh"

namespace rocgraph
{

    template rmm::device_uvector<int32_t>
        vertex_coloring(raft::handle_t const&                              handle,
                        graph_view_t<int32_t, int32_t, false, true> const& graph_view,
                        raft::random::RngState&                            rng_state);

    template rmm::device_uvector<int32_t>
        vertex_coloring(raft::handle_t const&                              handle,
                        graph_view_t<int32_t, int64_t, false, true> const& graph_view,
                        raft::random::RngState&                            rng_state);

    template rmm::device_uvector<int64_t>
        vertex_coloring(raft::handle_t const&                              handle,
                        graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                        raft::random::RngState&                            rng_state);

} // namespace rocgraph
