// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "traversal/k_hop_nbrs_impl.cuh"

namespace rocgraph
{

    // MG instantiation

    template std::tuple<rmm::device_uvector<size_t>, rmm::device_uvector<int32_t>>
        k_hop_nbrs(raft::handle_t const&                               handle,
                   graph_view_t<int32_t, int32_t, false, false> const& graph_view,
                   raft::device_span<int32_t const>                    start_vertices,
                   size_t                                              k,
                   bool                                                do_expensive_check);

    template std::tuple<rmm::device_uvector<size_t>, rmm::device_uvector<int32_t>>
        k_hop_nbrs(raft::handle_t const&                               handle,
                   graph_view_t<int32_t, int64_t, false, false> const& graph_view,
                   raft::device_span<int32_t const>                    start_vertices,
                   size_t                                              k,
                   bool                                                do_expensive_check);

    template std::tuple<rmm::device_uvector<size_t>, rmm::device_uvector<int64_t>>
        k_hop_nbrs(raft::handle_t const&                               handle,
                   graph_view_t<int64_t, int64_t, false, false> const& graph_view,
                   raft::device_span<int64_t const>                    start_vertices,
                   size_t                                              k,
                   bool                                                do_expensive_check);

} // namespace rocgraph
