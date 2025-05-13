// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "prims/fill_edge_property.cuh"

#include "utilities/misc_utils_device.hpp"

namespace rocgraph
{
    namespace c_api
    {

        template <typename vertex_t, typename edge_t>
        rmm::device_uvector<vertex_t> expand_sparse_offsets(raft::device_span<edge_t const> offsets,
                                                            vertex_t base_vertex_id,
                                                            rmm::cuda_stream_view const& stream)
        {
            return rocgraph::detail::expand_sparse_offsets(offsets, base_vertex_id, stream);
        }

        template <typename GraphViewType, typename T>
        edge_property_t<GraphViewType, T> create_constant_edge_property(
            raft::handle_t const& handle, GraphViewType const& graph_view, T constant_value)
        {
            edge_property_t<GraphViewType, T> edge_property(handle, graph_view);

            rocgraph::fill_edge_property(handle, graph_view, constant_value, edge_property);

            return edge_property;
        }

    } // namespace c_api
} // namespace rocgraph
