// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_abstract_functor.hpp"
#include "c_api/rocgraph_graph.hpp"
#include "c_api/rocgraph_handle.hpp"
#include "c_api/rocgraph_paths_result.hpp"
#include "c_api/rocgraph_utils.hpp"

#include "internal/aux/rocgraph_extract_paths_result_aux.h"
#include "internal/rocgraph_algorithms.h"

#include "algorithms.hpp"
#include "detail/utility_wrappers.hpp"
#include "graph_functions.hpp"

namespace rocgraph
{
    namespace c_api
    {

        struct rocgraph_extract_paths_result_t
        {
            size_t                               max_path_length_;
            rocgraph_type_erased_device_array_t* paths_;
        };

        struct extract_paths_functor : public abstract_functor
        {
            raft::handle_t const&                           handle_;
            rocgraph_graph_t*                               graph_;
            rocgraph_type_erased_device_array_view_t const* sources_;
            rocgraph_paths_result_t const*                  paths_result_;
            rocgraph_type_erased_device_array_view_t const* destinations_;
            rocgraph_extract_paths_result_t*                result_{};

            extract_paths_functor(::rocgraph_handle_t const*                        handle,
                                  ::rocgraph_graph_t*                               graph,
                                  ::rocgraph_type_erased_device_array_view_t const* sources,
                                  ::rocgraph_paths_result_t const*                  paths_result,
                                  ::rocgraph_type_erased_device_array_view_t const* destinations)
                : abstract_functor()
                , handle_(*handle->get_raft_handle())
                , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
                , sources_(reinterpret_cast<
                           rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                      sources))
                , paths_result_(reinterpret_cast<rocgraph::c_api::rocgraph_paths_result_t const*>(
                      paths_result))
                , destinations_(reinterpret_cast<
                                rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                      destinations))
            {
            }

            template <typename vertex_t,
                      typename edge_t,
                      typename weight_t,
                      typename edge_type_type_t,
                      bool store_transposed,
                      bool multi_gpu>
            void operator()()
            {
                // FIXME: Think about how to handle SG vice MG
                if constexpr(!rocgraph::is_candidate<vertex_t, edge_t, weight_t>::value)
                {
                    unsupported();
                }
                else
                {
                    // BFS and SSSP expect store_transposed == false
                    if constexpr(store_transposed)
                    {
                        status_ = rocgraph::c_api::transpose_storage<vertex_t,
                                                                     edge_t,
                                                                     weight_t,
                                                                     store_transposed,
                                                                     multi_gpu>(
                            handle_, graph_, error_.get());
                        if(status_ != rocgraph_status_success)
                            return;
                    }

                    auto graph
                        = reinterpret_cast<rocgraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(
                            graph_->graph_);

                    auto graph_view = graph->view();

                    auto number_map
                        = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

                    rmm::device_uvector<vertex_t> destinations(destinations_->size_,
                                                               handle_.get_stream());
                    raft::copy(destinations.data(),
                               destinations_->as_type<vertex_t>(),
                               destinations_->size_,
                               handle_.get_stream());

                    rmm::device_uvector<vertex_t> predecessors(paths_result_->predecessors_->size_,
                                                               handle_.get_stream());
                    raft::copy(predecessors.data(),
                               paths_result_->predecessors_->view()->as_type<vertex_t>(),
                               paths_result_->predecessors_->view()->size_,
                               handle_.get_stream());

                    //
                    // Need to renumber destinations
                    //
                    renumber_ext_vertices<vertex_t, multi_gpu>(
                        handle_,
                        destinations.data(),
                        destinations.size(),
                        number_map->data(),
                        graph_view.local_vertex_partition_range_first(),
                        graph_view.local_vertex_partition_range_last(),
                        false);

                    renumber_ext_vertices<vertex_t, multi_gpu>(
                        handle_,
                        predecessors.data(),
                        predecessors.size(),
                        number_map->data(),
                        graph_view.local_vertex_partition_range_first(),
                        graph_view.local_vertex_partition_range_last(),
                        false);

                    auto [result, max_path_length]
                        = rocgraph::extract_bfs_paths<vertex_t, edge_t, multi_gpu>(
                            handle_,
                            graph_view,
                            paths_result_->distances_->view()->as_type<vertex_t>(),
                            predecessors.data(),
                            destinations.data(),
                            destinations.size());

                    std::vector<vertex_t> vertex_partition_range_lasts
                        = graph_view.vertex_partition_range_lasts();

                    unrenumber_int_vertices<vertex_t, multi_gpu>(handle_,
                                                                 result.data(),
                                                                 result.size(),
                                                                 number_map->data(),
                                                                 vertex_partition_range_lasts,
                                                                 false);

                    result_ = new rocgraph_extract_paths_result_t{
                        static_cast<size_t>(max_path_length),
                        new rocgraph_type_erased_device_array_t(result, graph_->vertex_type_)};
                }
            }
        };

    } // namespace c_api
} // namespace rocgraph

extern "C" size_t
    rocgraph_extract_paths_result_get_max_path_length(rocgraph_extract_paths_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_extract_paths_result_t*>(result);
    return internal_pointer->max_path_length_;
}

rocgraph_type_erased_device_array_view_t*
    rocgraph_extract_paths_result_get_paths(rocgraph_extract_paths_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_extract_paths_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->paths_->view());
}

extern "C" void rocgraph_extract_paths_result_free(rocgraph_extract_paths_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_extract_paths_result_t*>(result);
    delete internal_pointer->paths_;
    delete internal_pointer;
}

extern "C" rocgraph_status
    rocgraph_extract_paths(const rocgraph_handle_t*                        handle,
                           rocgraph_graph_t*                               graph,
                           const rocgraph_type_erased_device_array_view_t* sources,
                           const rocgraph_paths_result_t*                  paths_result,
                           const rocgraph_type_erased_device_array_view_t* destinations,
                           rocgraph_extract_paths_result_t**               result,
                           rocgraph_error_t**                              error)
{
    rocgraph::c_api::extract_paths_functor functor(
        handle, graph, sources, paths_result, destinations);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}
