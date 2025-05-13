// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "community/legacy/louvain.cuh"

#include "community/flatten_dendrogram.hpp"

#include "graph.hpp"

#include <rmm/device_uvector.hpp>

#include <thrust/sequence.h>

namespace rocgraph
{

    namespace detail
    {

        template <typename vertex_t, typename edge_t, typename weight_t>
        void check_clustering(legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph_view,
                              vertex_t*                                               clustering)
        {
            ROCGRAPH_EXPECTS(clustering != nullptr, "Invalid input argument: clustering is null");
        }

        template <typename vertex_t, typename edge_t, typename weight_t>
        std::pair<std::unique_ptr<Dendrogram<vertex_t>>, weight_t>
            louvain(raft::handle_t const&                                   handle,
                    legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph_view,
                    size_t                                                  max_level,
                    weight_t                                                resolution)
        {
            ROCGRAPH_EXPECTS(graph_view.edge_data != nullptr,
                             "Invalid input argument: louvain expects a weighted graph");

            legacy::Louvain<legacy::GraphCSRView<vertex_t, edge_t, weight_t>> runner(handle,
                                                                                     graph_view);
            weight_t wt = runner(max_level, resolution);

            return std::make_pair(runner.move_dendrogram(), wt);
        }

        template <typename vertex_t, typename edge_t, typename weight_t>
        void flatten_dendrogram(raft::handle_t const&                                   handle,
                                legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph_view,
                                Dendrogram<vertex_t> const&                             dendrogram,
                                vertex_t*                                               clustering)
        {
            rmm::device_uvector<vertex_t> vertex_ids_v(graph_view.number_of_vertices,
                                                       handle.get_stream());

            thrust::sequence(
                handle.get_thrust_policy(), vertex_ids_v.begin(), vertex_ids_v.end(), vertex_t{0});

            partition_at_level<vertex_t, false>(
                handle, dendrogram, vertex_ids_v.data(), clustering, dendrogram.num_levels());
        }

    } // namespace detail

    template <typename vertex_t, typename edge_t, typename weight_t>
    std::pair<size_t, weight_t>
        louvain(raft::handle_t const&                                   handle,
                legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph_view,
                vertex_t*                                               clustering,
                size_t                                                  max_level,
                weight_t                                                resolution)
    {
        ROCGRAPH_EXPECTS(graph_view.has_data(), "Graph must be weighted");
        detail::check_clustering(graph_view, clustering);

        std::unique_ptr<Dendrogram<vertex_t>> dendrogram;
        weight_t                              modularity;

        std::tie(dendrogram, modularity)
            = detail::louvain(handle, graph_view, max_level, resolution);

        detail::flatten_dendrogram(handle, graph_view, *dendrogram, clustering);

        return std::make_pair(dendrogram->num_levels(), modularity);
    }

    // Explicit template instantations
    template std::pair<size_t, float> louvain(raft::handle_t const&,
                                              legacy::GraphCSRView<int32_t, int32_t, float> const&,
                                              int32_t*,
                                              size_t,
                                              float);
    template std::pair<size_t, double>
        louvain(raft::handle_t const&,
                legacy::GraphCSRView<int32_t, int32_t, double> const&,
                int32_t*,
                size_t,
                double);
} // namespace rocgraph
