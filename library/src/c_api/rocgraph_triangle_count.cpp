// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_abstract_functor.hpp"
#include "c_api/rocgraph_graph.hpp"
#include "c_api/rocgraph_handle.hpp"
#include "c_api/rocgraph_utils.hpp"

#include "internal/aux/rocgraph_triangle_count_result_aux.h"
#include "internal/rocgraph_algorithms.h"

#include "algorithms.hpp"
#include "detail/shuffle_wrappers.hpp"
#include "detail/utility_wrappers.hpp"
#include "graph_functions.hpp"

#include <optional>

namespace rocgraph
{
    namespace c_api
    {

        struct rocgraph_triangle_count_result_t
        {
            rocgraph_type_erased_device_array_t* vertices_;
            rocgraph_type_erased_device_array_t* counts_;
        };

    } // namespace c_api
} // namespace rocgraph

namespace
{

    struct triangle_count_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                                            handle_;
        rocgraph::c_api::rocgraph_graph_t*                               graph_;
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* vertices_;
        bool                                                             do_expensive_check_;
        rocgraph::c_api::rocgraph_triangle_count_result_t*               result_{};

        triangle_count_functor(::rocgraph_handle_t const*                        handle,
                               ::rocgraph_graph_t*                               graph,
                               ::rocgraph_type_erased_device_array_view_t const* vertices,
                               bool                                              do_expensive_check)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , vertices_(reinterpret_cast<
                        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(vertices))
            , do_expensive_check_(do_expensive_check)
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
            if constexpr(!rocgraph::is_candidate<vertex_t, edge_t, weight_t>::value)
            {
                unsupported();
            }
            else
            {
                // triangle counting expects store_transposed == false
                if constexpr(store_transposed)
                {
                    status_ = rocgraph::c_api::
                        transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
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

                rmm::device_uvector<vertex_t> vertices(0, handle_.get_stream());
                rmm::device_uvector<edge_t>   counts(0, handle_.get_stream());

                if(vertices_ != nullptr)
                {
                    vertices.resize(vertices_->size_, handle_.get_stream());

                    raft::copy(vertices.data(),
                               vertices_->as_type<vertex_t>(),
                               vertices.size(),
                               handle_.get_stream());

                    if constexpr(multi_gpu)
                    {
                        vertices = rocgraph::detail::
                            shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
                                handle_, std::move(vertices));
                    }

                    counts.resize(vertices.size(), handle_.get_stream());

                    rocgraph::renumber_ext_vertices<vertex_t, multi_gpu>(
                        handle_,
                        vertices.data(),
                        vertices.size(),
                        number_map->data(),
                        graph_view.local_vertex_partition_range_first(),
                        graph_view.local_vertex_partition_range_last(),
                        do_expensive_check_);
                }
                else
                {
                    counts.resize(graph_view.local_vertex_partition_range_size(),
                                  handle_.get_stream());
                }

                rocgraph::triangle_count<vertex_t, edge_t, multi_gpu>(
                    handle_,
                    graph_view,
                    vertices_ == nullptr ? std::nullopt
                                         : std::make_optional(raft::device_span<vertex_t>{
                                               vertices.data(), vertices.size()}),
                    raft::device_span<edge_t>{counts.data(), counts.size()},
                    do_expensive_check_);

                if(vertices_ == nullptr)
                {
                    vertices.resize(graph_view.local_vertex_partition_range_size(),
                                    handle_.get_stream());
                    raft::copy(
                        vertices.data(), number_map->data(), vertices.size(), handle_.get_stream());
                }
                else
                {
                    std::vector<vertex_t> vertex_partition_range_lasts
                        = graph_view.vertex_partition_range_lasts();

                    rocgraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
                        handle_,
                        vertices.data(),
                        vertices.size(),
                        number_map->data(),
                        vertex_partition_range_lasts,
                        do_expensive_check_);
                }

                result_ = new rocgraph::c_api::rocgraph_triangle_count_result_t{
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(vertices,
                                                                             graph_->vertex_type_),
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(counts,
                                                                             graph_->edge_type_)};
            }
        }
    };

} // namespace

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_triangle_count_result_get_vertices(rocgraph_triangle_count_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_triangle_count_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->vertices_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_triangle_count_result_get_counts(rocgraph_triangle_count_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_triangle_count_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->counts_->view());
}

extern "C" void rocgraph_triangle_count_result_free(rocgraph_triangle_count_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_triangle_count_result_t*>(result);
    delete internal_pointer->vertices_;
    delete internal_pointer->counts_;
    delete internal_pointer;
}

extern "C" rocgraph_status
    rocgraph_triangle_count(const rocgraph_handle_t*                        handle,
                            rocgraph_graph_t*                               graph,
                            const rocgraph_type_erased_device_array_view_t* start,
                            rocgraph_bool                                   do_expensive_check,
                            rocgraph_triangle_count_result_t**              result,
                            rocgraph_error_t**                              error)
{
    triangle_count_functor functor(handle, graph, start, do_expensive_check);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}
