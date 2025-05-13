// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_abstract_functor.hpp"
#include "c_api/rocgraph_graph.hpp"
#include "c_api/rocgraph_handle.hpp"
#include "c_api/rocgraph_utils.hpp"

#include "internal/aux/rocgraph_random_walk_result_aux.h"
#include "internal/rocgraph_algorithms.h"

#include "algorithms.hpp"
#include "detail/utility_wrappers.hpp"
#include "graph_functions.hpp"

namespace rocgraph
{
    namespace c_api
    {

        struct rocgraph_random_walk_result_t
        {
            bool                                 result_compressed_{false};
            size_t                               max_path_length_{0};
            rocgraph_type_erased_device_array_t* paths_{nullptr};
            rocgraph_type_erased_device_array_t* weights_{nullptr};
            rocgraph_type_erased_device_array_t* sizes_{nullptr};
        };

        struct node2vec_functor : public abstract_functor
        {
            raft::handle_t const&                           handle_;
            rocgraph_graph_t*                               graph_{nullptr};
            rocgraph_type_erased_device_array_view_t const* start_vertices_{nullptr};
            size_t                                          max_length_{0};
            bool                                            compress_result_{false};
            double                                          p_{0};
            double                                          q_{0};
            rocgraph_random_walk_result_t*                  result_{nullptr};

            node2vec_functor(::rocgraph_handle_t const*                        handle,
                             ::rocgraph_graph_t*                               graph,
                             ::rocgraph_type_erased_device_array_view_t const* start_vertices,
                             size_t                                            max_length,
                             bool                                              compress_result,
                             double                                            p,
                             double                                            q)
                : abstract_functor()
                , handle_(*handle->get_raft_handle())
                , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
                , start_vertices_(reinterpret_cast<
                                  rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                      start_vertices))
                , max_length_(max_length)
                , compress_result_(compress_result)
                , p_(p)
                , q_(q)
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
                else if constexpr(multi_gpu)
                {
                    unsupported();
                }
                else
                {
                    // node2vec expects store_transposed == false
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

                    auto edge_weights = reinterpret_cast<rocgraph::edge_property_t<
                        rocgraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                        weight_t>*>(graph_->edge_weights_);

                    auto number_map
                        = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

                    rmm::device_uvector<vertex_t> start_vertices(start_vertices_->size_,
                                                                 handle_.get_stream());
                    raft::copy(start_vertices.data(),
                               start_vertices_->as_type<vertex_t>(),
                               start_vertices.size(),
                               handle_.get_stream());

                    //
                    // Need to renumber start_vertices
                    //
                    renumber_ext_vertices<vertex_t, multi_gpu>(
                        handle_,
                        start_vertices.data(),
                        start_vertices.size(),
                        number_map->data(),
                        graph_view.local_vertex_partition_range_first(),
                        graph_view.local_vertex_partition_range_last(),
                        false);

                    // FIXME:  Forcing this to edge_t for now.  What should it really be?
                    // Seems like it should be the smallest size that can accommodate
                    // max_length_ * start_vertices_->size_
                    auto [paths, weights, sizes] = rocgraph::random_walks(
                        handle_,
                        graph_view,
                        (edge_weights != nullptr) ? std::make_optional(edge_weights->view())
                                                  : std::nullopt,
                        start_vertices.data(),
                        static_cast<edge_t>(start_vertices.size()),
                        static_cast<edge_t>(max_length_),
                        !compress_result_,
                        // std::make_unique<sampling_params_t>(2, p_, q_, false));
                        std::make_unique<sampling_params_t>(
                            rocgraph::sampling_strategy_t::NODE2VEC, p_, q_));

                    //
                    // Need to unrenumber the vertices in the resulting paths
                    //
                    unrenumber_local_int_vertices<vertex_t>(handle_,
                                                            paths.data(),
                                                            paths.size(),
                                                            number_map->data(),
                                                            0,
                                                            paths.size() - 1,
                                                            false);

                    result_ = new rocgraph_random_walk_result_t{
                        compress_result_,
                        max_length_,
                        new rocgraph_type_erased_device_array_t(paths, graph_->vertex_type_),
                        new rocgraph_type_erased_device_array_t(weights, graph_->weight_type_),
                        new rocgraph_type_erased_device_array_t(sizes, graph_->vertex_type_)};
                }
            }
        };

    } // namespace c_api
} // namespace rocgraph

namespace
{

    struct uniform_random_walks_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                                            handle_;
        rocgraph::c_api::rocgraph_graph_t*                               graph_{nullptr};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* start_vertices_{nullptr};
        size_t                                                           max_length_{0};
        size_t                                                           seed_{0};
        rocgraph::c_api::rocgraph_random_walk_result_t*                  result_{nullptr};

        uniform_random_walks_functor(rocgraph_handle_t const*                        handle,
                                     rocgraph_graph_t*                               graph,
                                     rocgraph_type_erased_device_array_view_t const* start_vertices,
                                     size_t                                          max_length)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , start_vertices_(reinterpret_cast<
                              rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                  start_vertices))
            , max_length_(max_length)
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
                // random walks expects store_transposed == false
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

                rmm::device_uvector<vertex_t> start_vertices(start_vertices_->size_,
                                                             handle_.get_stream());
                raft::copy(start_vertices.data(),
                           start_vertices_->as_type<vertex_t>(),
                           start_vertices.size(),
                           handle_.get_stream());

                //
                // Need to renumber start_vertices
                //
                rocgraph::renumber_ext_vertices<vertex_t, multi_gpu>(
                    handle_,
                    start_vertices.data(),
                    start_vertices.size(),
                    number_map->data(),
                    graph_view.local_vertex_partition_range_first(),
                    graph_view.local_vertex_partition_range_last(),
                    false);

                auto [paths, weights] = rocgraph::uniform_random_walks(
                    handle_,
                    graph_view,
                    (edge_weights != nullptr) ? std::make_optional(edge_weights->view())
                                              : std::nullopt,
                    raft::device_span<vertex_t const>{start_vertices.data(), start_vertices.size()},
                    max_length_,
                    seed_);

                //
                // Need to unrenumber the vertices in the resulting paths
                //
                rocgraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
                    handle_,
                    paths.data(),
                    paths.size(),
                    number_map->data(),
                    graph_view.vertex_partition_range_lasts(),
                    false);

                result_ = new rocgraph::c_api::rocgraph_random_walk_result_t{
                    false,
                    max_length_,
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(paths,
                                                                             graph_->vertex_type_),
                    weights ? new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                                  *weights, graph_->weight_type_)
                            : nullptr,
                    nullptr};
            }
        }
    };

    struct biased_random_walks_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                                            handle_;
        rocgraph::c_api::rocgraph_graph_t*                               graph_{nullptr};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* start_vertices_{nullptr};
        size_t                                                           max_length_{0};
        rocgraph::c_api::rocgraph_random_walk_result_t*                  result_{nullptr};
        uint64_t                                                         seed_{0};

        biased_random_walks_functor(rocgraph_handle_t const*                        handle,
                                    rocgraph_graph_t*                               graph,
                                    rocgraph_type_erased_device_array_view_t const* start_vertices,
                                    size_t                                          max_length)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , start_vertices_(reinterpret_cast<
                              rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                  start_vertices))
            , max_length_(max_length)
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
            else if constexpr(multi_gpu)
            {
                unsupported();
            }
            else
            {
                // random walks expects store_transposed == false
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

                rmm::device_uvector<vertex_t> start_vertices(start_vertices_->size_,
                                                             handle_.get_stream());
                raft::copy(start_vertices.data(),
                           start_vertices_->as_type<vertex_t>(),
                           start_vertices.size(),
                           handle_.get_stream());

                //
                // Need to renumber start_vertices
                //
                rocgraph::renumber_ext_vertices<vertex_t, multi_gpu>(
                    handle_,
                    start_vertices.data(),
                    start_vertices.size(),
                    number_map->data(),
                    graph_view.local_vertex_partition_range_first(),
                    graph_view.local_vertex_partition_range_last(),
                    false);

                auto [paths, weights] = rocgraph::biased_random_walks(
                    handle_,
                    graph_view,
                    edge_weights->view(),
                    raft::device_span<vertex_t const>{start_vertices.data(), start_vertices.size()},
                    max_length_,
                    seed_);

                //
                // Need to unrenumber the vertices in the resulting paths
                //
                rocgraph::unrenumber_local_int_vertices<vertex_t>(handle_,
                                                                  paths.data(),
                                                                  paths.size(),
                                                                  number_map->data(),
                                                                  0,
                                                                  paths.size() - 1,
                                                                  false);

                result_ = new rocgraph::c_api::rocgraph_random_walk_result_t{
                    false,
                    max_length_,
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(paths,
                                                                             graph_->vertex_type_),
                    weights ? new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                                  *weights, graph_->weight_type_)
                            : nullptr,
                    nullptr};
            }
        }
    };

    struct node2vec_random_walks_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                                            handle_;
        rocgraph::c_api::rocgraph_graph_t*                               graph_{nullptr};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* start_vertices_{nullptr};
        size_t                                                           max_length_{0};
        double                                                           p_{0};
        double                                                           q_{0};
        uint64_t                                                         seed_{0};
        rocgraph::c_api::rocgraph_random_walk_result_t*                  result_{nullptr};

        node2vec_random_walks_functor(
            rocgraph_handle_t const*                        handle,
            rocgraph_graph_t*                               graph,
            rocgraph_type_erased_device_array_view_t const* start_vertices,
            size_t                                          max_length,
            double                                          p,
            double                                          q)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , start_vertices_(reinterpret_cast<
                              rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                  start_vertices))
            , max_length_(max_length)
            , p_(p)
            , q_(q)
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
            else if constexpr(multi_gpu)
            {
                unsupported();
            }
            else
            {
                // random walks expects store_transposed == false
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

                rmm::device_uvector<vertex_t> start_vertices(start_vertices_->size_,
                                                             handle_.get_stream());
                raft::copy(start_vertices.data(),
                           start_vertices_->as_type<vertex_t>(),
                           start_vertices.size(),
                           handle_.get_stream());

                //
                // Need to renumber start_vertices
                //
                rocgraph::renumber_ext_vertices<vertex_t, multi_gpu>(
                    handle_,
                    start_vertices.data(),
                    start_vertices.size(),
                    number_map->data(),
                    graph_view.local_vertex_partition_range_first(),
                    graph_view.local_vertex_partition_range_last(),
                    false);

                auto [paths, weights] = rocgraph::node2vec_random_walks(
                    handle_,
                    graph_view,
                    (edge_weights != nullptr) ? std::make_optional(edge_weights->view())
                                              : std::nullopt,
                    raft::device_span<vertex_t const>{start_vertices.data(), start_vertices.size()},
                    max_length_,
                    static_cast<weight_t>(p_),
                    static_cast<weight_t>(q_),
                    seed_);

                // FIXME:  Need to fix invalid_vtx issue here.  We can't unrenumber max_vertex_id+1
                // properly...
                //   So if the result includes an invalid vertex we don't handle it properly.

                //
                // Need to unrenumber the vertices in the resulting paths
                //
                rocgraph::unrenumber_local_int_vertices<vertex_t>(handle_,
                                                                  paths.data(),
                                                                  paths.size(),
                                                                  number_map->data(),
                                                                  0,
                                                                  paths.size(),
                                                                  false);

                result_ = new rocgraph::c_api::rocgraph_random_walk_result_t{
                    false,
                    max_length_,
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(paths,
                                                                             graph_->vertex_type_),
                    weights ? new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                                  *weights, graph_->weight_type_)
                            : nullptr,
                    nullptr};
            }
        }
    };

} // anonymous namespace

rocgraph_status rocgraph_node2vec(const rocgraph_handle_t*                        handle,
                                  rocgraph_graph_t*                               graph,
                                  const rocgraph_type_erased_device_array_view_t* start_vertices,
                                  size_t                                          max_length,
                                  rocgraph_bool                                   compress_results,
                                  double                                          p,
                                  double                                          q,
                                  rocgraph_random_walk_result_t**                 result,
                                  rocgraph_error_t**                              error)
{
    CAPI_EXPECTS(
        reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->vertex_type_
            == reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                   start_vertices)
                   ->type_,
        rocgraph_status_invalid_input,
        "vertex type of graph and start_vertices must match",
        *error);

    rocgraph::c_api::node2vec_functor functor(
        handle, graph, start_vertices, max_length, compress_results, p, q);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}

size_t rocgraph_random_walk_result_get_max_path_length(rocgraph_random_walk_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_random_walk_result_t*>(result);
    return internal_pointer->max_path_length_;
}

rocgraph_type_erased_device_array_view_t*
    rocgraph_random_walk_result_get_paths(rocgraph_random_walk_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_random_walk_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->paths_->view());
}

rocgraph_type_erased_device_array_view_t*
    rocgraph_random_walk_result_get_weights(rocgraph_random_walk_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_random_walk_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->weights_->view());
}

rocgraph_type_erased_device_array_view_t*
    rocgraph_random_walk_result_get_path_sizes(rocgraph_random_walk_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_random_walk_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->sizes_->view());
}

void rocgraph_random_walk_result_free(rocgraph_random_walk_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_random_walk_result_t*>(result);
    delete internal_pointer->paths_;
    delete internal_pointer->sizes_;
    delete internal_pointer->weights_;
    delete internal_pointer;
}

rocgraph_status
    rocgraph_uniform_random_walks(const rocgraph_handle_t*                        handle,
                                  rocgraph_graph_t*                               graph,
                                  const rocgraph_type_erased_device_array_view_t* start_vertices,
                                  size_t                                          max_length,
                                  rocgraph_random_walk_result_t**                 result,
                                  rocgraph_error_t**                              error)
{
    CAPI_EXPECTS(
        reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->vertex_type_
            == reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                   start_vertices)
                   ->type_,
        rocgraph_status_invalid_input,
        "vertex type of graph and start_vertices must match",
        *error);

    uniform_random_walks_functor functor(handle, graph, start_vertices, max_length);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}

rocgraph_status
    rocgraph_biased_random_walks(const rocgraph_handle_t*                        handle,
                                 rocgraph_graph_t*                               graph,
                                 const rocgraph_type_erased_device_array_view_t* start_vertices,
                                 size_t                                          max_length,
                                 rocgraph_random_walk_result_t**                 result,
                                 rocgraph_error_t**                              error)
{
    CAPI_EXPECTS(
        reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->vertex_type_
            == reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                   start_vertices)
                   ->type_,
        rocgraph_status_invalid_input,
        "vertex type of graph and start_vertices must match",
        *error);

    biased_random_walks_functor functor(handle, graph, start_vertices, max_length);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}

rocgraph_status
    rocgraph_node2vec_random_walks(const rocgraph_handle_t*                        handle,
                                   rocgraph_graph_t*                               graph,
                                   const rocgraph_type_erased_device_array_view_t* start_vertices,
                                   size_t                                          max_length,
                                   double                                          p,
                                   double                                          q,
                                   rocgraph_random_walk_result_t**                 result,
                                   rocgraph_error_t**                              error)
{
    CAPI_EXPECTS(
        reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->vertex_type_
            == reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                   start_vertices)
                   ->type_,
        rocgraph_status_invalid_input,
        "vertex type of graph and start_vertices must match",
        *error);

    node2vec_random_walks_functor functor(handle, graph, start_vertices, max_length, p, q);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}
