// Copyright (C) 2020-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "algorithms.hpp"
#include "legacy/graph.hpp"
#include "utilities/error.hpp"

#include <raft/sparse/solver/mst.cuh>

#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/transform.h>

#include <ctime>
#include <memory>
#include <utility>

namespace rocgraph
{

    namespace detail
    {

        template <typename vertex_t, typename edge_t, typename weight_t>
        std::unique_ptr<legacy::GraphCOO<vertex_t, edge_t, weight_t>>
            mst_impl(raft::handle_t const&                                   handle,
                     legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                     rmm::device_async_resource_ref                          mr)

        {
            auto                          stream = handle.get_stream();
            rmm::device_uvector<vertex_t> colors(graph.number_of_vertices, stream);
            auto                          mst_edges
                = raft::sparse::solver::mst<vertex_t, edge_t, weight_t>(handle,
                                                                        graph.offsets,
                                                                        graph.indices,
                                                                        graph.edge_data,
                                                                        graph.number_of_vertices,
                                                                        graph.number_of_edges,
                                                                        colors.data(),
                                                                        stream);

            legacy::GraphCOOContents<vertex_t, edge_t, weight_t> coo_contents{
                graph.number_of_vertices,
                mst_edges.n_edges,
                std::make_unique<rmm::device_buffer>(mst_edges.src.release()),
                std::make_unique<rmm::device_buffer>(mst_edges.dst.release()),
                std::make_unique<rmm::device_buffer>(mst_edges.weights.release())};

            return std::make_unique<legacy::GraphCOO<vertex_t, edge_t, weight_t>>(
                std::move(coo_contents));
        }

    } // namespace detail

    template <typename vertex_t, typename edge_t, typename weight_t>
    std::unique_ptr<legacy::GraphCOO<vertex_t, edge_t, weight_t>>
        minimum_spanning_tree(raft::handle_t const&                                   handle,
                              legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                              rmm::device_async_resource_ref                          mr)
    {
        return detail::mst_impl(handle, graph, mr);
    }

    template std::unique_ptr<legacy::GraphCOO<int, int, float>>
        minimum_spanning_tree<int, int, float>(raft::handle_t const&                        handle,
                                               legacy::GraphCSRView<int, int, float> const& graph,
                                               rmm::device_async_resource_ref               mr);
    template std::unique_ptr<legacy::GraphCOO<int, int, double>>
        minimum_spanning_tree<int, int, double>(raft::handle_t const& handle,
                                                legacy::GraphCSRView<int, int, double> const& graph,
                                                rmm::device_async_resource_ref                mr);
} // namespace rocgraph
