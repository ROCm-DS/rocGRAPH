// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "components/weakly_connected_components_impl.cuh"

namespace rocgraph
{

    // SG instantiations

    template void
        weakly_connected_components(raft::handle_t const&                               handle,
                                    graph_view_t<int32_t, int32_t, false, false> const& graph_view,
                                    int32_t*                                            components,
                                    bool do_expensive_check);

    template void
        weakly_connected_components(raft::handle_t const&                               handle,
                                    graph_view_t<int32_t, int64_t, false, false> const& graph_view,
                                    int32_t*                                            components,
                                    bool do_expensive_check);

    template void
        weakly_connected_components(raft::handle_t const&                               handle,
                                    graph_view_t<int64_t, int64_t, false, false> const& graph_view,
                                    int64_t*                                            components,
                                    bool do_expensive_check);

} // namespace rocgraph
