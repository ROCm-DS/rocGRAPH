// Copyright (C) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#include "edge_property.hpp"
#include "graph_view.hpp"

#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

namespace rocgraph
{
    namespace detail
    {

        template <typename vertex_t, typename edge_t, bool multi_gpu>
        rmm::device_uvector<vertex_t> maximal_independent_moves(
            raft::handle_t const&                                   handle,
            graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
            raft::random::RngState&                                 rng_state);

    } // namespace detail
} // namespace rocgraph
