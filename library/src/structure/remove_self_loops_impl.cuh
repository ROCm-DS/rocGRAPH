// Copyright (C) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "structure/detail/structure_utils.cuh"

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <optional>

namespace rocgraph
{

    template <typename vertex_t, typename edge_t, typename weight_t, typename edge_type_t>
    void remove_self_loops_helper(
        raft::handle_t const&                            handle,
        rmm::device_uvector<vertex_t>&                   edgelist_srcs,
        rmm::device_uvector<vertex_t>&                   edgelist_dsts,
        std::optional<rmm::device_uvector<weight_t>>&    edgelist_weights,
        std::optional<rmm::device_uvector<edge_t>>&      edgelist_edge_ids,
        std::optional<rmm::device_uvector<edge_type_t>>& edgelist_edge_types)
    {

        auto [keep_count, keep_flags] = detail::mark_entries(
            handle,
            edgelist_srcs.size(),
            [d_srcs = edgelist_srcs.data(), d_dsts = edgelist_dsts.data()] __device__(size_t i) {
                return d_srcs[i] != d_dsts[i];
            });

        if(keep_count < edgelist_srcs.size())
        {
            edgelist_srcs
                = detail::keep_flagged_elements(handle,
                                                std::move(edgelist_srcs),
                                                raft::device_span<packed_bool_container_t const>{
                                                    keep_flags.data(), keep_flags.size()},
                                                keep_count);
            edgelist_dsts
                = detail::keep_flagged_elements(handle,
                                                std::move(edgelist_dsts),
                                                raft::device_span<packed_bool_container_t const>{
                                                    keep_flags.data(), keep_flags.size()},
                                                keep_count);

            if(edgelist_weights)
                edgelist_weights = detail::keep_flagged_elements(
                    handle,
                    std::move(*edgelist_weights),
                    raft::device_span<packed_bool_container_t const>{keep_flags.data(),
                                                                     keep_flags.size()},
                    keep_count);

            if(edgelist_edge_ids)
                edgelist_edge_ids = detail::keep_flagged_elements(
                    handle,
                    std::move(*edgelist_edge_ids),
                    raft::device_span<packed_bool_container_t const>{keep_flags.data(),
                                                                     keep_flags.size()},
                    keep_count);

            if(edgelist_edge_types)
                edgelist_edge_types = detail::keep_flagged_elements(
                    handle,
                    std::move(*edgelist_edge_types),
                    raft::device_span<packed_bool_container_t const>{keep_flags.data(),
                                                                     keep_flags.size()},
                    keep_count);
        }
    }

    template <typename vertex_t, typename edge_t, typename weight_t, typename edge_type_t>
    std::tuple<rmm::device_uvector<vertex_t>,
               rmm::device_uvector<vertex_t>,
               std::optional<rmm::device_uvector<weight_t>>,
               std::optional<rmm::device_uvector<edge_t>>,
               std::optional<rmm::device_uvector<edge_type_t>>>
        remove_self_loops(raft::handle_t const&                             handle,
                          rmm::device_uvector<vertex_t>&&                   edgelist_srcs,
                          rmm::device_uvector<vertex_t>&&                   edgelist_dsts,
                          std::optional<rmm::device_uvector<weight_t>>&&    edgelist_weights,
                          std::optional<rmm::device_uvector<edge_t>>&&      edgelist_edge_ids,
                          std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types)
    {

        remove_self_loops_helper(handle,
                                 edgelist_srcs,
                                 edgelist_dsts,
                                 edgelist_weights,
                                 edgelist_edge_ids,
                                 edgelist_edge_types);

        return std::make_tuple(std::move(edgelist_srcs),
                               std::move(edgelist_dsts),
                               std::move(edgelist_weights),
                               std::move(edgelist_edge_ids),
                               std::move(edgelist_edge_types));
    }

} // namespace rocgraph
