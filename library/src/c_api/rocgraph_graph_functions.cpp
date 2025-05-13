// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_graph_functions.hpp"

#include "c_api/rocgraph_abstract_functor.hpp"
#include "c_api/rocgraph_graph.hpp"
#include "c_api/rocgraph_graph_helper.hpp"
#include "c_api/rocgraph_handle.hpp"
#include "c_api/rocgraph_utils.hpp"

#include "algorithms.hpp"
#include "detail/shuffle_wrappers.hpp"
#include "detail/utility_wrappers.hpp"
#include "graph_functions.hpp"
#include "internal/aux/rocgraph_vertex_pairs_aux.h"
#include "internal/rocgraph_graph_functions.h"

namespace
{

    struct create_vertex_pairs_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                                            handle_;
        rocgraph::c_api::rocgraph_graph_t*                               graph_;
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* first_;
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* second_;
        rocgraph::c_api::rocgraph_vertex_pairs_t*                        result_{};

        create_vertex_pairs_functor(::rocgraph_handle_t const*                        handle,
                                    ::rocgraph_graph_t*                               graph,
                                    ::rocgraph_type_erased_device_array_view_t const* first,
                                    ::rocgraph_type_erased_device_array_view_t const* second)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , first_(reinterpret_cast<
                     rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(first))
            , second_(reinterpret_cast<
                      rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(second))
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
                rmm::device_uvector<vertex_t> first_copy(first_->size_, handle_.get_stream());
                rmm::device_uvector<vertex_t> second_copy(second_->size_, handle_.get_stream());

                raft::copy(first_copy.data(),
                           first_->as_type<vertex_t>(),
                           first_->size_,
                           handle_.get_stream());
                raft::copy(second_copy.data(),
                           second_->as_type<vertex_t>(),
                           second_->size_,
                           handle_.get_stream());

                if constexpr(multi_gpu)
                {
                    std::tie(first_copy, second_copy, std::ignore, std::ignore, std::ignore)
                        = rocgraph::detail::
                            shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<
                                vertex_t,
                                edge_t,
                                weight_t,
                                edge_type_type_t>(handle_,
                                                  std::move(first_copy),
                                                  std::move(second_copy),
                                                  std::nullopt,
                                                  std::nullopt,
                                                  std::nullopt);
                }

                result_ = new rocgraph::c_api::rocgraph_vertex_pairs_t{
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(first_copy,
                                                                             graph_->vertex_type_),
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(second_copy,
                                                                             graph_->vertex_type_)};
            }
        }
    };

    struct two_hop_neighbors_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                                            handle_{};
        rocgraph::c_api::rocgraph_graph_t*                               graph_{nullptr};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* start_vertices_{nullptr};
        rocgraph::c_api::rocgraph_vertex_pairs_t*                        result_{};
        bool                                                             do_expensive_check_{false};

        two_hop_neighbors_functor(::rocgraph_handle_t const*                        handle,
                                  ::rocgraph_graph_t*                               graph,
                                  ::rocgraph_type_erased_device_array_view_t const* start_vertices,
                                  bool do_expensive_check)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , start_vertices_(reinterpret_cast<
                              rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                  start_vertices))
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
                // k_hop_nbrs expects store_transposed == false
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

                rmm::device_uvector<vertex_t> start_vertices(0, handle_.get_stream());

                if(start_vertices_ != nullptr)
                {
                    start_vertices.resize(start_vertices_->size_, handle_.get_stream());
                    raft::copy(start_vertices.data(),
                               start_vertices_->as_type<vertex_t const>(),
                               start_vertices_->size_,
                               handle_.get_stream());

                    if constexpr(multi_gpu)
                    {
                        start_vertices = rocgraph::detail::
                            shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
                                handle_, std::move(start_vertices));
                    }

                    rocgraph::renumber_ext_vertices<vertex_t, multi_gpu>(
                        handle_,
                        start_vertices.data(),
                        start_vertices.size(),
                        number_map->data(),
                        graph_view.local_vertex_partition_range_first(),
                        graph_view.local_vertex_partition_range_last(),
                        do_expensive_check_);
                }
                else
                {
                    start_vertices.resize(graph_view.local_vertex_partition_range_size(),
                                          handle_.get_stream());
                    rocgraph::detail::sequence_fill(
                        handle_.get_stream(),
                        start_vertices.data(),
                        start_vertices.size(),
                        graph_view.local_vertex_partition_range_first());
                }

                auto [offsets, dst] = rocgraph::k_hop_nbrs(
                    handle_,
                    graph_view,
                    raft::device_span<vertex_t const>{start_vertices.data(), start_vertices.size()},
                    size_t{2},
                    do_expensive_check_);

                auto src = rocgraph::c_api::expand_sparse_offsets(
                    raft::device_span<size_t const>{offsets.data(), offsets.size()},
                    vertex_t{0},
                    handle_.get_stream());

                // convert ids back to srcs:  src[i] = start_vertices[src[i]]
                rocgraph::unrenumber_local_int_vertices(
                    handle_,
                    src.data(),
                    src.size(),
                    start_vertices.data(),
                    vertex_t{0},
                    graph_view.local_vertex_partition_range_size(),
                    do_expensive_check_);

                rocgraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
                    handle_,
                    src.data(),
                    src.size(),
                    number_map->data(),
                    graph_view.vertex_partition_range_lasts(),
                    do_expensive_check_);

                rocgraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
                    handle_,
                    dst.data(),
                    dst.size(),
                    number_map->data(),
                    graph_view.vertex_partition_range_lasts(),
                    do_expensive_check_);

                result_ = new rocgraph::c_api::rocgraph_vertex_pairs_t{
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(src,
                                                                             graph_->vertex_type_),
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(dst,
                                                                             graph_->vertex_type_)};
            }
        }
    };

} // namespace

extern "C" rocgraph_status
    rocgraph_create_vertex_pairs(const rocgraph_handle_t*                        handle,
                                 rocgraph_graph_t*                               graph,
                                 const rocgraph_type_erased_device_array_view_t* first,
                                 const rocgraph_type_erased_device_array_view_t* second,
                                 rocgraph_vertex_pairs_t**                       vertex_pairs,
                                 rocgraph_error_t**                              error)
{
    CAPI_EXPECTS(
        reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->vertex_type_
            == reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                   first)
                   ->type_,
        rocgraph_status_invalid_input,
        "vertex type of graph and first must match",
        *error);

    CAPI_EXPECTS(
        reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->vertex_type_
            == reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                   second)
                   ->type_,
        rocgraph_status_invalid_input,
        "vertex type of graph and second must match",
        *error);

    create_vertex_pairs_functor functor(handle, graph, first, second);

    return rocgraph::c_api::run_algorithm(graph, functor, vertex_pairs, error);
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_vertex_pairs_get_first(rocgraph_vertex_pairs_t* vertex_pairs)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_vertex_pairs_t*>(vertex_pairs);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->first_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_vertex_pairs_get_second(rocgraph_vertex_pairs_t* vertex_pairs)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_vertex_pairs_t*>(vertex_pairs);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->second_->view());
}

extern "C" void rocgraph_vertex_pairs_free(rocgraph_vertex_pairs_t* vertex_pairs)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_vertex_pairs_t*>(vertex_pairs);
    delete internal_pointer->first_;
    delete internal_pointer->second_;
    delete internal_pointer;
}

extern "C" rocgraph_status
    rocgraph_two_hop_neighbors(const rocgraph_handle_t*                        handle,
                               rocgraph_graph_t*                               graph,
                               const rocgraph_type_erased_device_array_view_t* start_vertices,
                               rocgraph_bool                                   do_expensive_check,
                               rocgraph_vertex_pairs_t**                       result,
                               rocgraph_error_t**                              error)
{
    two_hop_neighbors_functor functor(handle, graph, start_vertices, do_expensive_check);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}
