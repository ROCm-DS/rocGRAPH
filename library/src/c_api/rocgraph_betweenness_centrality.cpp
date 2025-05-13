// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_abstract_functor.hpp"
#include "c_api/rocgraph_centrality_result.hpp"
#include "c_api/rocgraph_graph.hpp"
#include "c_api/rocgraph_handle.hpp"
#include "c_api/rocgraph_random.hpp"
#include "c_api/rocgraph_utils.hpp"

#include "internal/rocgraph_algorithms.h"

#include "algorithms.hpp"
#include "detail/shuffle_wrappers.hpp"
#include "detail/utility_wrappers.hpp"
#include "graph_functions.hpp"

#include <optional>

namespace
{

    struct betweenness_centrality_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                                            handle_;
        rocgraph::c_api::rocgraph_graph_t*                               graph_{};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* vertex_list_{};
        rocgraph_bool                                                    normalized_{};
        rocgraph_bool                                                    include_endpoints_{};
        bool                                                             do_expensive_check_{};
        rocgraph::c_api::rocgraph_centrality_result_t*                   result_{};

        betweenness_centrality_functor(rocgraph_handle_t const*                        handle,
                                       rocgraph_graph_t*                               graph,
                                       rocgraph_type_erased_device_array_view_t const* vertex_list,
                                       rocgraph_bool                                   normalized,
                                       rocgraph_bool include_endpoints,
                                       bool          do_expensive_check)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , vertex_list_(reinterpret_cast<
                           rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                  vertex_list))
            , normalized_(normalized)
            , include_endpoints_(include_endpoints)
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
            // FIXME: Think about how to handle SG vice MG
            if constexpr(!rocgraph::is_candidate<vertex_t, edge_t, weight_t>::value)
            {
                unsupported();
            }
            else
            {
                // Betweenness Centrality expects store_transposed == false
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

                auto edge_weights = reinterpret_cast<rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                    weight_t>*>(graph_->edge_weights_);

                auto number_map
                    = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

                rmm::device_uvector<vertex_t> local_vertices(0, handle_.get_stream());

                std::optional<raft::device_span<vertex_t const>> vertex_span{std::nullopt};
                if(vertex_list_ != nullptr)
                {
                    local_vertices.resize(vertex_list_->size_, handle_.get_stream());
                    raft::copy(local_vertices.data(),
                               vertex_list_->as_type<vertex_t>(),
                               vertex_list_->size_,
                               handle_.get_stream());

                    if constexpr(multi_gpu)
                    {
                        local_vertices = rocgraph::detail::
                            shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
                                handle_, std::move(local_vertices));
                    }

                    rocgraph::renumber_ext_vertices<vertex_t, multi_gpu>(
                        handle_,
                        local_vertices.data(),
                        local_vertices.size(),
                        number_map->data(),
                        graph_view.local_vertex_partition_range_first(),
                        graph_view.local_vertex_partition_range_last(),
                        do_expensive_check_);

                    vertex_span = raft::device_span<vertex_t const>{local_vertices.data(),
                                                                    local_vertices.size()};
                }

                auto centralities
                    = rocgraph::betweenness_centrality<vertex_t, edge_t, weight_t, multi_gpu>(
                        handle_,
                        graph_view,
                        (edge_weights != nullptr) ? std::make_optional(edge_weights->view())
                                                  : std::nullopt,
                        vertex_span,
                        normalized_,
                        include_endpoints_,
                        do_expensive_check_);

                rmm::device_uvector<vertex_t> vertex_ids(
                    graph_view.local_vertex_partition_range_size(), handle_.get_stream());
                raft::copy(
                    vertex_ids.data(), number_map->data(), vertex_ids.size(), handle_.get_stream());

                result_ = new rocgraph::c_api::rocgraph_centrality_result_t{
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(vertex_ids,
                                                                             graph_->vertex_type_),
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(centralities,
                                                                             graph_->weight_type_)};
            }
        }
    };

    struct edge_betweenness_centrality_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                                            handle_;
        rocgraph::c_api::rocgraph_graph_t*                               graph_{};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* vertex_list_{};
        rocgraph_bool                                                    normalized_{};
        bool                                                             do_expensive_check_{};
        rocgraph::c_api::rocgraph_edge_centrality_result_t*              result_{};

        edge_betweenness_centrality_functor(
            rocgraph_handle_t const*                        handle,
            rocgraph_graph_t*                               graph,
            rocgraph_type_erased_device_array_view_t const* vertex_list,
            rocgraph_bool                                   normalized,
            bool                                            do_expensive_check)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , vertex_list_(reinterpret_cast<
                           rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                  vertex_list))
            , normalized_(normalized)
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
            // FIXME: Think about how to handle SG vice MG
            if constexpr(!rocgraph::is_candidate<vertex_t, edge_t, weight_t>::value)
            {
                unsupported();
            }
            else
            {
                // Betweenness Centrality expects store_transposed == false
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

                auto edge_weights = reinterpret_cast<rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                    weight_t>*>(graph_->edge_weights_);

                auto edge_ids = reinterpret_cast<rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                    edge_t>*>(graph_->edge_ids_);

                auto number_map
                    = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

                rmm::device_uvector<vertex_t> local_vertices(0, handle_.get_stream());

                std::optional<raft::device_span<vertex_t const>> vertex_span{std::nullopt};
                if(vertex_list_ != nullptr)
                {
                    local_vertices.resize(vertex_list_->size_, handle_.get_stream());
                    raft::copy(local_vertices.data(),
                               vertex_list_->as_type<vertex_t>(),
                               vertex_list_->size_,
                               handle_.get_stream());

                    if constexpr(multi_gpu)
                    {
                        local_vertices = rocgraph::detail::
                            shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
                                handle_, std::move(local_vertices));
                    }

                    rocgraph::renumber_ext_vertices<vertex_t, multi_gpu>(
                        handle_,
                        local_vertices.data(),
                        local_vertices.size(),
                        number_map->data(),
                        graph_view.local_vertex_partition_range_first(),
                        graph_view.local_vertex_partition_range_last(),
                        do_expensive_check_);

                    vertex_span = raft::device_span<vertex_t const>{local_vertices.data(),
                                                                    local_vertices.size()};
                }

                auto centralities
                    = rocgraph::edge_betweenness_centrality<vertex_t, edge_t, weight_t, multi_gpu>(
                        handle_,
                        graph_view,
                        (edge_weights != nullptr) ? std::make_optional(edge_weights->view())
                                                  : std::nullopt,
                        vertex_span,
                        normalized_,
                        do_expensive_check_);

                auto [src_ids, dst_ids, output_centralities, output_edge_ids, output_edge_types]
                    = rocgraph::decompress_to_edgelist<vertex_t,
                                                       edge_t,
                                                       weight_t,
                                                       int32_t,
                                                       false,
                                                       multi_gpu>(
                        handle_,
                        graph_view,
                        std::make_optional(centralities.view()),
                        (edge_ids != nullptr) ? std::make_optional(edge_ids->view()) : std::nullopt,
                        std::nullopt,
                        (number_map != nullptr)
                            ? std::make_optional(raft::device_span<vertex_t const>{
                                  number_map->data(), number_map->size()})
                            : std::nullopt);

                result_ = new rocgraph::c_api::rocgraph_edge_centrality_result_t{
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(src_ids,
                                                                             graph_->vertex_type_),
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(dst_ids,
                                                                             graph_->vertex_type_),
                    output_edge_ids ? new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                                          *output_edge_ids, graph_->edge_type_)
                                    : nullptr,
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(*output_centralities,
                                                                             graph_->weight_type_)};
            }
        }
    };

} // namespace

extern "C" rocgraph_status
    rocgraph_betweenness_centrality(const rocgraph_handle_t*                        handle,
                                    rocgraph_graph_t*                               graph,
                                    const rocgraph_type_erased_device_array_view_t* vertex_list,
                                    rocgraph_bool                                   normalized,
                                    rocgraph_bool                  include_endpoints,
                                    rocgraph_bool                  do_expensive_check,
                                    rocgraph_centrality_result_t** result,
                                    rocgraph_error_t**             error)
{
    betweenness_centrality_functor functor(
        handle, graph, vertex_list, normalized, include_endpoints, do_expensive_check);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}

extern "C" rocgraph_status rocgraph_edge_betweenness_centrality(
    const rocgraph_handle_t*                        handle,
    rocgraph_graph_t*                               graph,
    const rocgraph_type_erased_device_array_view_t* vertex_list,
    rocgraph_bool                                   normalized,
    rocgraph_bool                                   do_expensive_check,
    rocgraph_edge_centrality_result_t**             result,
    rocgraph_error_t**                              error)
{
    edge_betweenness_centrality_functor functor(
        handle, graph, vertex_list, normalized, do_expensive_check);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}
