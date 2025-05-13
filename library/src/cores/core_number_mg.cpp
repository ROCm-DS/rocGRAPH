// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "cores/core_number_impl.cuh"

namespace rocgraph
{

    // MG instantiation

    template void core_number(raft::handle_t const&                              handle,
                              graph_view_t<int32_t, int32_t, false, true> const& graph_view,
                              int32_t*                                           core_numbers,
                              k_core_degree_type_t                               degree_type,
                              size_t                                             k_first,
                              size_t                                             k_last,
                              bool do_expensive_check);

    template void core_number(raft::handle_t const&                              handle,
                              graph_view_t<int32_t, int64_t, false, true> const& graph_view,
                              int64_t*                                           core_numbers,
                              k_core_degree_type_t                               degree_type,
                              size_t                                             k_first,
                              size_t                                             k_last,
                              bool do_expensive_check);

    template void core_number(raft::handle_t const&                              handle,
                              graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                              int64_t*                                           core_numbers,
                              k_core_degree_type_t                               degree_type,
                              size_t                                             k_first,
                              size_t                                             k_last,
                              bool do_expensive_check);

} // namespace rocgraph
