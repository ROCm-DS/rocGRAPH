// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_abstract_functor.hpp"
#include "c_api/rocgraph_graph.hpp"
#include "c_api/rocgraph_handle.hpp"
#include "c_api/rocgraph_paths_result.hpp"
#include "c_api/rocgraph_utils.hpp"

#include "internal/aux/rocgraph_paths_result_aux.h"
#include "internal/rocgraph_algorithms.h"

#include "algorithms.hpp"
#include "detail/shuffle_wrappers.hpp"
#include "detail/utility_wrappers.hpp"
#include "graph_functions.hpp"

namespace rocgraph
{
    namespace c_api
    {

        struct bfs_functor : public abstract_functor
        {
            raft::handle_t const&                     handle_;
            rocgraph_graph_t*                         graph_;
            rocgraph_type_erased_device_array_view_t* sources_;
            bool                                      direction_optimizing_;
            size_t                                    depth_limit_;
            bool                                      compute_predecessors_;
            bool                                      do_expensive_check_;
            rocgraph_paths_result_t*                  result_{};

            bfs_functor(::rocgraph_handle_t const*                  handle,
                        ::rocgraph_graph_t*                         graph,
                        ::rocgraph_type_erased_device_array_view_t* sources,
                        bool                                        direction_optimizing,
                        size_t                                      depth_limit,
                        bool                                        compute_predecessors,
                        bool                                        do_expensive_check)
                : abstract_functor()
                , handle_(*handle->get_raft_handle())
                , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
                , sources_(
                      reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t*>(
                          sources))
                , direction_optimizing_(direction_optimizing)
                , depth_limit_(depth_limit)
                , compute_predecessors_(compute_predecessors)
                , do_expensive_check_(do_expensive_check)
            {
            }

            template <typename vertex_t,
                      typename edge_t,
                      typename weight_t,
                      typename edge_type_t,
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
                    // BFS expects store_transposed == false
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

                    rmm::device_uvector<vertex_t> distances(
                        graph_view.local_vertex_partition_range_size(), handle_.get_stream());
                    rmm::device_uvector<vertex_t> predecessors(0, handle_.get_stream());

                    if(compute_predecessors_)
                    {
                        predecessors.resize(graph_view.local_vertex_partition_range_size(),
                                            handle_.get_stream());
                    }

                    rmm::device_uvector<vertex_t> sources(sources_->size_, handle_.get_stream());
                    raft::copy(sources.data(),
                               sources_->as_type<vertex_t>(),
                               sources_->size_,
                               handle_.get_stream());

                    if constexpr(multi_gpu)
                    {
                        sources = detail::shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
                            handle_, std::move(sources));
                    }

                    //
                    // Need to renumber sources
                    //
                    renumber_ext_vertices<vertex_t, multi_gpu>(
                        handle_,
                        sources.data(),
                        sources.size(),
                        number_map->data(),
                        graph_view.local_vertex_partition_range_first(),
                        graph_view.local_vertex_partition_range_last(),
                        do_expensive_check_);

                    size_t invalid_count = rocgraph::detail::count_values(
                        handle_,
                        raft::device_span<vertex_t const>{sources.data(), sources.size()},
                        rocgraph::invalid_vertex_id<vertex_t>::value);

                    if constexpr(multi_gpu)
                    {
                        invalid_count = rocgraph::host_scalar_allreduce(handle_.get_comms(),
                                                                        invalid_count,
                                                                        raft::comms::op_t::SUM,
                                                                        handle_.get_stream());
                    }

                    if(invalid_count != 0)
                    {
                        mark_error(rocgraph_status_invalid_input,
                                   "Found invalid vertex in the input sources");
                        return;
                    }

                    rocgraph::bfs<vertex_t, edge_t, multi_gpu>(
                        handle_,
                        graph_view,
                        distances.data(),
                        compute_predecessors_ ? predecessors.data() : nullptr,
                        sources.data(),
                        sources.size(),
                        direction_optimizing_,
                        static_cast<vertex_t>(depth_limit_),
                        do_expensive_check_);

                    rmm::device_uvector<vertex_t> vertex_ids(
                        graph_view.local_vertex_partition_range_size(), handle_.get_stream());
                    raft::copy(vertex_ids.data(),
                               number_map->data(),
                               vertex_ids.size(),
                               handle_.get_stream());

                    if(compute_predecessors_)
                    {
                        std::vector<vertex_t> vertex_partition_range_lasts
                            = graph_view.vertex_partition_range_lasts();

                        unrenumber_int_vertices<vertex_t, multi_gpu>(handle_,
                                                                     predecessors.data(),
                                                                     predecessors.size(),
                                                                     number_map->data(),
                                                                     vertex_partition_range_lasts,
                                                                     do_expensive_check_);
                    }

                    result_ = new rocgraph_paths_result_t{
                        new rocgraph_type_erased_device_array_t(vertex_ids, graph_->vertex_type_),
                        new rocgraph_type_erased_device_array_t(distances, graph_->vertex_type_),
                        new rocgraph_type_erased_device_array_t(predecessors,
                                                                graph_->vertex_type_)};
                }
            }
        };

    } // namespace c_api
} // namespace rocgraph

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_paths_result_get_vertices(rocgraph_paths_result_t* result)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_paths_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->vertex_ids_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_paths_result_get_distances(rocgraph_paths_result_t* result)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_paths_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->distances_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_paths_result_get_predecessors(rocgraph_paths_result_t* result)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_paths_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->predecessors_->view());
}

extern "C" void rocgraph_paths_result_free(rocgraph_paths_result_t* result)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_paths_result_t*>(result);
    delete internal_pointer->vertex_ids_;
    delete internal_pointer->distances_;
    delete internal_pointer->predecessors_;
    delete internal_pointer;
}

extern "C" rocgraph_status rocgraph_bfs(const rocgraph_handle_t*                  handle,
                                        rocgraph_graph_t*                         graph,
                                        rocgraph_type_erased_device_array_view_t* sources,
                                        rocgraph_bool             direction_optimizing,
                                        size_t                    depth_limit,
                                        rocgraph_bool             compute_predecessors,
                                        rocgraph_bool             do_expensive_check,
                                        rocgraph_paths_result_t** result,
                                        rocgraph_error_t**        error)
{
    CAPI_EXPECTS(
        reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->vertex_type_
            == reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                   sources)
                   ->type_,
        rocgraph_status_invalid_input,
        "vertex type of graph and sources must match",
        *error);

    rocgraph::c_api::bfs_functor functor(handle,
                                         graph,
                                         sources,
                                         direction_optimizing,
                                         depth_limit,
                                         compute_predecessors,
                                         do_expensive_check);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}
