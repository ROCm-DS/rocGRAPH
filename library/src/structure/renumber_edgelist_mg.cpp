// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "structure/renumber_edgelist_impl.cuh"

namespace rocgraph
{

    // MG instantiation

    template std::tuple<rmm::device_uvector<int32_t>, renumber_meta_t<int32_t, int32_t, true>>
        renumber_edgelist<int32_t, int32_t, true>(
            raft::handle_t const&                         handle,
            std::optional<rmm::device_uvector<int32_t>>&& local_vertices,
            std::vector<int32_t*> const&                  edgelist_srcs /* [INOUT] */,
            std::vector<int32_t*> const&                  edgelist_dsts /* [INOUT] */,
            std::vector<int32_t> const&                   edgelist_edge_counts,
            std::optional<std::vector<std::vector<int32_t>>> const&
                 edgelist_intra_partition_segment_offsets,
            bool store_transposed,
            bool do_expensive_check);

    template std::tuple<rmm::device_uvector<int32_t>, renumber_meta_t<int32_t, int64_t, true>>
        renumber_edgelist<int32_t, int64_t, true>(
            raft::handle_t const&                         handle,
            std::optional<rmm::device_uvector<int32_t>>&& local_vertices,
            std::vector<int32_t*> const&                  edgelist_srcs /* [INOUT] */,
            std::vector<int32_t*> const&                  edgelist_dsts /* [INOUT] */,
            std::vector<int64_t> const&                   edgelist_edge_counts,
            std::optional<std::vector<std::vector<int64_t>>> const&
                 edgelist_intra_partition_segment_offsets,
            bool store_transposed,
            bool do_expensive_check);

    template std::tuple<rmm::device_uvector<int64_t>, renumber_meta_t<int64_t, int64_t, true>>
        renumber_edgelist<int64_t, int64_t, true>(
            raft::handle_t const&                         handle,
            std::optional<rmm::device_uvector<int64_t>>&& local_vertices,
            std::vector<int64_t*> const&                  edgelist_srcs /* [INOUT] */,
            std::vector<int64_t*> const&                  edgelist_dsts /* [INOUT] */,
            std::vector<int64_t> const&                   edgelist_edge_counts,
            std::optional<std::vector<std::vector<int64_t>>> const&
                 edgelist_intra_partition_segment_offsets,
            bool store_transposed,
            bool do_expensive_check);

} // namespace rocgraph
