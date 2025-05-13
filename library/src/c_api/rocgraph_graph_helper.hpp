// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>

namespace rocgraph
{
    namespace c_api
    {

        template <typename vertex_t, typename edge_t>
        rmm::device_uvector<vertex_t> expand_sparse_offsets(raft::device_span<edge_t const> offsets,
                                                            vertex_t base_vertex_id,
                                                            rmm::cuda_stream_view const& stream);

        template <typename GraphViewType, typename T>
        edge_property_t<GraphViewType, T> create_constant_edge_property(
            raft::handle_t const& handle, GraphViewType const& graph_view, T constant_value);

    } // namespace c_api
} // namespace rocgraph
