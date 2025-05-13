// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "sampling/detail/prepare_next_frontier_impl.cuh"

namespace rocgraph
{
    namespace detail
    {

        template std::tuple<rmm::device_uvector<int32_t>,
                            std::optional<rmm::device_uvector<int32_t>>,
                            std::optional<std::tuple<rmm::device_uvector<int32_t>,
                                                     std::optional<rmm::device_uvector<int32_t>>>>>
            prepare_next_frontier(
                raft::handle_t const&                           handle,
                raft::device_span<int32_t const>                sampled_src_vertices,
                std::optional<raft::device_span<int32_t const>> sampled_src_vertex_labels,
                raft::device_span<int32_t const>                sampled_dst_vertices,
                std::optional<raft::device_span<int32_t const>> sampled_dst_vertex_labels,
                std::optional<std::tuple<rmm::device_uvector<int32_t>,
                                         std::optional<rmm::device_uvector<int32_t>>>>&&
                                                        vertex_used_as_source,
                vertex_partition_view_t<int32_t, false> vertex_partition,
                std::vector<int32_t> const&             vertex_partition_range_lasts,
                prior_sources_behavior_t                prior_sources_behavior,
                bool                                    dedupe_sources,
                bool                                    do_expensive_check);

        template std::tuple<rmm::device_uvector<int64_t>,
                            std::optional<rmm::device_uvector<int32_t>>,
                            std::optional<std::tuple<rmm::device_uvector<int64_t>,
                                                     std::optional<rmm::device_uvector<int32_t>>>>>
            prepare_next_frontier(
                raft::handle_t const&                           handle,
                raft::device_span<int64_t const>                sampled_src_vertices,
                std::optional<raft::device_span<int32_t const>> sampled_src_vertex_labels,
                raft::device_span<int64_t const>                sampled_dst_vertices,
                std::optional<raft::device_span<int32_t const>> sampled_dst_vertex_labels,
                std::optional<std::tuple<rmm::device_uvector<int64_t>,
                                         std::optional<rmm::device_uvector<int32_t>>>>&&
                                                        vertex_used_as_source,
                vertex_partition_view_t<int64_t, false> vertex_partition,
                std::vector<int64_t> const&             vertex_partition_range_lasts,
                prior_sources_behavior_t                prior_sources_behavior,
                bool                                    dedupe_sources,
                bool                                    do_expensive_check);

    } // namespace detail
} // namespace rocgraph
