// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "structure/renumber_edgelist_impl.cuh"

namespace rocgraph
{

    // SG instantiation

    template std::tuple<rmm::device_uvector<int32_t>, renumber_meta_t<int32_t, int32_t, false>>
        renumber_edgelist<int32_t, int32_t, false>(
            raft::handle_t const&                         handle,
            std::optional<rmm::device_uvector<int32_t>>&& vertices,
            int32_t*                                      edgelist_srcs /* [INOUT] */,
            int32_t*                                      edgelist_dsts /* [INOUT] */,
            int32_t                                       num_edgelist_edges,
            bool                                          store_transposed,
            bool                                          do_expensive_check);

    template std::tuple<rmm::device_uvector<int32_t>, renumber_meta_t<int32_t, int64_t, false>>
        renumber_edgelist<int32_t, int64_t, false>(
            raft::handle_t const&                         handle,
            std::optional<rmm::device_uvector<int32_t>>&& vertices,
            int32_t*                                      edgelist_srcs /* [INOUT] */,
            int32_t*                                      edgelist_dsts /* [INOUT] */,
            int64_t                                       num_edgelist_edges,
            bool                                          store_transposed,
            bool                                          do_expensive_check);

    template std::tuple<rmm::device_uvector<int64_t>, renumber_meta_t<int64_t, int64_t, false>>
        renumber_edgelist<int64_t, int64_t, false>(
            raft::handle_t const&                         handle,
            std::optional<rmm::device_uvector<int64_t>>&& vertices,
            int64_t*                                      edgelist_srcs /* [INOUT] */,
            int64_t*                                      edgelist_dsts /* [INOUT] */,
            int64_t                                       num_edgelist_edges,
            bool                                          store_transposed,
            bool                                          do_expensive_check);

} // namespace rocgraph
