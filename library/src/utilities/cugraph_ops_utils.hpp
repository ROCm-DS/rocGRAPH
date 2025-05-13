// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "graph_view.hpp"
#include <rocgraph-ops/graph/format.hpp>

namespace rocgraph
{
    namespace detail
    {

        template <typename NodeTypeT, typename EdgeTypeT>
        ops::graph::csc<EdgeTypeT, NodeTypeT>
            get_graph(graph_view_t<NodeTypeT, EdgeTypeT, false, false> const& gview)
        {
            ops::graph::csc<EdgeTypeT, NodeTypeT> graph;
            graph.n_src_nodes = gview.number_of_vertices();
            graph.n_dst_nodes = gview.number_of_vertices();
            graph.n_indices   = gview.local_edge_partition_view().number_of_edges();
            // FIXME this is sufficient for now, but if there is a fast (cached) way
            // of getting max degree, use that instead
            graph.dst_max_in_degree = std::numeric_limits<EdgeTypeT>::max();
            // FIXME: this is evil and is just temporary until we have a matching type in rocgraph-ops
            // or we change the type accepted by the functions calling into rocgraph-ops
            graph.offsets
                = const_cast<EdgeTypeT*>(gview.local_edge_partition_view().offsets().data());
            graph.indices
                = const_cast<EdgeTypeT*>(gview.local_edge_partition_view().indices().data());
            return graph;
        }

    } // namespace detail
} // namespace rocgraph
