// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <tuple>

namespace rocgraph
{
    namespace c_api
    {
        namespace detail
        {

            template <typename vertex_t>
            std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<size_t>>
                shuffle_vertex_ids_and_offsets(raft::handle_t const&           handle,
                                               rmm::device_uvector<vertex_t>&& vertices,
                                               raft::device_span<size_t const> offsets);

            template <typename key_t, typename value_t>
            void sort_by_key(raft::handle_t const&      handle,
                             raft::device_span<key_t>   keys,
                             raft::device_span<value_t> values);

            template <typename vertex_t, typename weight_t>
            std::tuple<rmm::device_uvector<size_t>,
                       rmm::device_uvector<vertex_t>,
                       rmm::device_uvector<vertex_t>,
                       std::optional<rmm::device_uvector<weight_t>>>
                reorder_extracted_egonets(
                    raft::handle_t const&                          handle,
                    rmm::device_uvector<size_t>&&                  source_indices,
                    rmm::device_uvector<size_t>&&                  offsets,
                    rmm::device_uvector<vertex_t>&&                edge_srcs,
                    rmm::device_uvector<vertex_t>&&                edge_dsts,
                    std::optional<rmm::device_uvector<weight_t>>&& edge_weights);

        } // namespace detail
    } // namespace c_api
} // namespace rocgraph
