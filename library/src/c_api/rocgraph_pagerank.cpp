// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_abstract_functor.hpp"
#include "c_api/rocgraph_centrality_result.hpp"
#include "c_api/rocgraph_graph.hpp"
#include "c_api/rocgraph_handle.hpp"
#include "c_api/rocgraph_utils.hpp"

#include "internal/aux/rocgraph_centrality_result_aux.h"
#include "internal/rocgraph_algorithms.h"

#include "algorithms.hpp"
#include "detail/shuffle_wrappers.hpp"
#include "detail/utility_wrappers.hpp"
#include "graph_functions.hpp"

#include <optional>

namespace
{

    struct pagerank_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&              handle_;
        rocgraph::c_api::rocgraph_graph_t* graph_{};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*
            precomputed_vertex_out_weight_vertices_{};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*
            precomputed_vertex_out_weight_sums_{};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* initial_guess_vertices_{};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* initial_guess_values_{};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*
            personalization_vertices_{};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* personalization_values_{};
        double                                                           alpha_{};
        double                                                           epsilon_{};
        size_t                                                           max_iterations_{};
        bool                                                             do_expensive_check_{};
        rocgraph::c_api::rocgraph_centrality_result_t*                   result_{};

        pagerank_functor(
            rocgraph_handle_t const*                        handle,
            rocgraph_graph_t*                               graph,
            rocgraph_type_erased_device_array_view_t const* precomputed_vertex_out_weight_vertices,
            rocgraph_type_erased_device_array_view_t const* precomputed_vertex_out_weight_sums,
            rocgraph_type_erased_device_array_view_t const* initial_guess_vertices,
            rocgraph_type_erased_device_array_view_t const* initial_guess_values,
            rocgraph_type_erased_device_array_view_t const* personalization_vertices,
            rocgraph_type_erased_device_array_view_t const* personalization_values,
            double                                          alpha,
            double                                          epsilon,
            size_t                                          max_iterations,
            bool                                            do_expensive_check)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , precomputed_vertex_out_weight_vertices_(
                  reinterpret_cast<
                      rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                      precomputed_vertex_out_weight_vertices))
            , precomputed_vertex_out_weight_sums_(
                  reinterpret_cast<
                      rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                      precomputed_vertex_out_weight_sums))
            , initial_guess_vertices_(
                  reinterpret_cast<
                      rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                      initial_guess_vertices))
            , initial_guess_values_(
                  reinterpret_cast<
                      rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                      initial_guess_values))
            , personalization_vertices_(
                  reinterpret_cast<
                      rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                      personalization_vertices))
            , personalization_values_(
                  reinterpret_cast<
                      rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                      personalization_values))
            , alpha_(alpha)
            , epsilon_(epsilon)
            , max_iterations_(max_iterations)
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
                // Pagerank expects store_transposed == true
                if constexpr(!store_transposed)
                {
                    status_ = rocgraph::c_api::
                        transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
                            handle_, graph_, error_.get());
                    if(status_ != rocgraph_status_success)
                        return;
                }

                auto graph
                    = reinterpret_cast<rocgraph::graph_t<vertex_t, edge_t, true, multi_gpu>*>(
                        graph_->graph_);

                auto graph_view = graph->view();

                auto edge_weights = reinterpret_cast<rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, true, multi_gpu>,
                    weight_t>*>(graph_->edge_weights_);

                auto number_map
                    = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

                rmm::device_uvector<weight_t> initial_pageranks(0, handle_.get_stream());
                rmm::device_uvector<vertex_t> personalization_vertices(0, handle_.get_stream());
                rmm::device_uvector<weight_t> personalization_values(0, handle_.get_stream());

                if(personalization_vertices_ != nullptr)
                {
                    personalization_vertices.resize(personalization_vertices_->size_,
                                                    handle_.get_stream());
                    personalization_values.resize(personalization_values_->size_,
                                                  handle_.get_stream());

                    raft::copy(personalization_vertices.data(),
                               personalization_vertices_->as_type<vertex_t>(),
                               personalization_vertices_->size_,
                               handle_.get_stream());
                    raft::copy(personalization_values.data(),
                               personalization_values_->as_type<weight_t>(),
                               personalization_values_->size_,
                               handle_.get_stream());

                    if constexpr(multi_gpu)
                    {
                        std::tie(personalization_vertices, personalization_values)
                            = rocgraph::detail::
                                shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
                                    handle_,
                                    std::move(personalization_vertices),
                                    std::move(personalization_values));
                    }
                    //
                    // Need to renumber personalization_vertices
                    //
                    rocgraph::renumber_local_ext_vertices<vertex_t, multi_gpu>(
                        handle_,
                        personalization_vertices.data(),
                        personalization_vertices.size(),
                        number_map->data(),
                        graph_view.local_vertex_partition_range_first(),
                        graph_view.local_vertex_partition_range_last(),
                        do_expensive_check_);
                }

                rmm::device_uvector<weight_t> precomputed_vertex_out_weight_sums(
                    0, handle_.get_stream());
                if(precomputed_vertex_out_weight_sums_ != nullptr)
                {
                    rmm::device_uvector<vertex_t> precomputed_vertex_out_weight_vertices(
                        precomputed_vertex_out_weight_vertices_->size_, handle_.get_stream());
                    precomputed_vertex_out_weight_sums.resize(
                        precomputed_vertex_out_weight_sums_->size_, handle_.get_stream());

                    raft::copy(precomputed_vertex_out_weight_vertices.data(),
                               precomputed_vertex_out_weight_vertices_->as_type<vertex_t>(),
                               precomputed_vertex_out_weight_vertices_->size_,
                               handle_.get_stream());
                    raft::copy(precomputed_vertex_out_weight_sums.data(),
                               precomputed_vertex_out_weight_sums_->as_type<weight_t>(),
                               precomputed_vertex_out_weight_sums_->size_,
                               handle_.get_stream());

                    precomputed_vertex_out_weight_sums
                        = rocgraph::detail::collect_local_vertex_values_from_ext_vertex_value_pairs<
                            vertex_t,
                            weight_t,
                            multi_gpu>(handle_,
                                       std::move(precomputed_vertex_out_weight_vertices),
                                       std::move(precomputed_vertex_out_weight_sums),
                                       *number_map,
                                       graph_view.local_vertex_partition_range_first(),
                                       graph_view.local_vertex_partition_range_last(),
                                       weight_t{0},
                                       do_expensive_check_);
                }

                if(initial_guess_values_ != nullptr)
                {
                    rmm::device_uvector<vertex_t> initial_guess_vertices(
                        initial_guess_vertices_->size_, handle_.get_stream());
                    rmm::device_uvector<weight_t> initial_guess_values(initial_guess_values_->size_,
                                                                       handle_.get_stream());

                    raft::copy(initial_guess_vertices.data(),
                               initial_guess_vertices_->as_type<vertex_t>(),
                               initial_guess_vertices.size(),
                               handle_.get_stream());

                    raft::copy(initial_guess_values.data(),
                               initial_guess_values_->as_type<weight_t>(),
                               initial_guess_values.size(),
                               handle_.get_stream());

                    initial_pageranks
                        = rocgraph::detail::collect_local_vertex_values_from_ext_vertex_value_pairs<
                            vertex_t,
                            weight_t,
                            multi_gpu>(handle_,
                                       std::move(initial_guess_vertices),
                                       std::move(initial_guess_values),
                                       *number_map,
                                       graph_view.local_vertex_partition_range_first(),
                                       graph_view.local_vertex_partition_range_last(),
                                       weight_t{0},
                                       do_expensive_check_);
                }

                auto [pageranks, metadata]
                    = rocgraph::pagerank<vertex_t, edge_t, weight_t, weight_t, multi_gpu>(
                        handle_,
                        graph_view,
                        (edge_weights != nullptr) ? std::make_optional(edge_weights->view())
                                                  : std::nullopt,
                        precomputed_vertex_out_weight_sums_
                            ? std::make_optional(raft::device_span<weight_t const>{
                                  precomputed_vertex_out_weight_sums.data(),
                                  precomputed_vertex_out_weight_sums.size()})
                            : std::nullopt,
                        personalization_vertices_
                            ? std::make_optional(std::make_tuple(
                                  raft::device_span<vertex_t const>{
                                      personalization_vertices.data(),
                                      personalization_vertices.size()},
                                  raft::device_span<weight_t const>{personalization_values.data(),
                                                                    personalization_values.size()}))
                            : std::nullopt,
                        initial_guess_values_ != nullptr
                            ? std::make_optional(raft::device_span<weight_t const>{
                                  initial_pageranks.data(), initial_pageranks.size()})
                            : std::nullopt,
                        static_cast<weight_t>(alpha_),
                        static_cast<weight_t>(epsilon_),
                        max_iterations_,
                        do_expensive_check_);

                rmm::device_uvector<vertex_t> vertex_ids(
                    graph_view.local_vertex_partition_range_size(), handle_.get_stream());
                raft::copy(
                    vertex_ids.data(), number_map->data(), vertex_ids.size(), handle_.get_stream());

                result_ = new rocgraph::c_api::rocgraph_centrality_result_t{
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(vertex_ids,
                                                                             graph_->vertex_type_),
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(pageranks,
                                                                             graph_->weight_type_),
                    metadata.number_of_iterations_,
                    metadata.converged_};
            }
        }
    };

} // namespace

extern "C" rocgraph_status rocgraph_pagerank(
    const rocgraph_handle_t*                        handle,
    rocgraph_graph_t*                               graph,
    const rocgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices,
    const rocgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
    const rocgraph_type_erased_device_array_view_t* initial_guess_vertices,
    const rocgraph_type_erased_device_array_view_t* initial_guess_values,
    double                                          alpha,
    double                                          epsilon,
    size_t                                          max_iterations,
    rocgraph_bool                                   do_expensive_check,
    rocgraph_centrality_result_t**                  result,
    rocgraph_error_t**                              error)
{

    if(precomputed_vertex_out_weight_vertices != nullptr)
    {
        CAPI_EXPECTS(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->vertex_type_
                         == reinterpret_cast<
                                rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                                precomputed_vertex_out_weight_vertices)
                                ->type_,
                     rocgraph_status_invalid_input,
                     "vertex type of graph and precomputed_vertex_out_weight_vertices must match",
                     *error);
        CAPI_EXPECTS(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->weight_type_
                         == reinterpret_cast<
                                rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                                precomputed_vertex_out_weight_sums)
                                ->type_,
                     rocgraph_status_invalid_input,
                     "vertex type of graph and precomputed_vertex_out_weight_sums must match",
                     *error);
    }
    if(initial_guess_vertices != nullptr)
    {
        CAPI_EXPECTS(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->vertex_type_
                         == reinterpret_cast<
                                rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                                initial_guess_vertices)
                                ->type_,
                     rocgraph_status_invalid_input,
                     "vertex type of graph and initial_guess_vertices must match",
                     *error);
        CAPI_EXPECTS(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->weight_type_
                         == reinterpret_cast<
                                rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                                initial_guess_values)
                                ->type_,
                     rocgraph_status_invalid_input,
                     "vertex type of graph and initial_guess_values must match",
                     *error);
    }
    pagerank_functor functor(handle,
                             graph,
                             precomputed_vertex_out_weight_vertices,
                             precomputed_vertex_out_weight_sums,
                             initial_guess_vertices,
                             initial_guess_values,
                             nullptr,
                             nullptr,
                             alpha,
                             epsilon,
                             max_iterations,
                             do_expensive_check);

    auto return_value = rocgraph::c_api::run_algorithm(graph, functor, result, error);

    CAPI_EXPECTS(rocgraph_centrality_result_converged(*result) == rocgraph_bool_true,
                 rocgraph_status_unknown_error,
                 "PageRank failed to converge.",
                 *error);

    return return_value;
}

extern "C" rocgraph_status rocgraph_pagerank_allow_nonconvergence(
    const rocgraph_handle_t*                        handle,
    rocgraph_graph_t*                               graph,
    const rocgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices,
    const rocgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
    const rocgraph_type_erased_device_array_view_t* initial_guess_vertices,
    const rocgraph_type_erased_device_array_view_t* initial_guess_values,
    double                                          alpha,
    double                                          epsilon,
    size_t                                          max_iterations,
    rocgraph_bool                                   do_expensive_check,
    rocgraph_centrality_result_t**                  result,
    rocgraph_error_t**                              error)
{
    if(precomputed_vertex_out_weight_vertices != nullptr)
    {
        CAPI_EXPECTS(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->vertex_type_
                         == reinterpret_cast<
                                rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                                precomputed_vertex_out_weight_vertices)
                                ->type_,
                     rocgraph_status_invalid_input,
                     "vertex type of graph and precomputed_vertex_out_weight_vertices must match",
                     *error);
        CAPI_EXPECTS(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->weight_type_
                         == reinterpret_cast<
                                rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                                precomputed_vertex_out_weight_sums)
                                ->type_,
                     rocgraph_status_invalid_input,
                     "vertex type of graph and precomputed_vertex_out_weight_sums must match",
                     *error);
    }
    if(initial_guess_vertices != nullptr)
    {
        CAPI_EXPECTS(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->vertex_type_
                         == reinterpret_cast<
                                rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                                initial_guess_vertices)
                                ->type_,
                     rocgraph_status_invalid_input,
                     "vertex type of graph and initial_guess_vertices must match",
                     *error);
        CAPI_EXPECTS(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->weight_type_
                         == reinterpret_cast<
                                rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                                initial_guess_values)
                                ->type_,
                     rocgraph_status_invalid_input,
                     "vertex type of graph and initial_guess_values must match",
                     *error);
    }
    pagerank_functor functor(handle,
                             graph,
                             precomputed_vertex_out_weight_vertices,
                             precomputed_vertex_out_weight_sums,
                             initial_guess_vertices,
                             initial_guess_values,
                             nullptr,
                             nullptr,
                             alpha,
                             epsilon,
                             max_iterations,
                             do_expensive_check);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}

extern "C" rocgraph_status rocgraph_personalized_pagerank(
    const rocgraph_handle_t*                        handle,
    rocgraph_graph_t*                               graph,
    const rocgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices,
    const rocgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
    const rocgraph_type_erased_device_array_view_t* initial_guess_vertices,
    const rocgraph_type_erased_device_array_view_t* initial_guess_values,
    const rocgraph_type_erased_device_array_view_t* personalization_vertices,
    const rocgraph_type_erased_device_array_view_t* personalization_values,
    double                                          alpha,
    double                                          epsilon,
    size_t                                          max_iterations,
    rocgraph_bool                                   do_expensive_check,
    rocgraph_centrality_result_t**                  result,
    rocgraph_error_t**                              error)
{
    if(precomputed_vertex_out_weight_vertices != nullptr)
    {
        CAPI_EXPECTS(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->vertex_type_
                         == reinterpret_cast<
                                rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                                precomputed_vertex_out_weight_vertices)
                                ->type_,
                     rocgraph_status_invalid_input,
                     "vertex type of graph and precomputed_vertex_out_weight_vertices must match",
                     *error);
        CAPI_EXPECTS(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->weight_type_
                         == reinterpret_cast<
                                rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                                precomputed_vertex_out_weight_sums)
                                ->type_,
                     rocgraph_status_invalid_input,
                     "vertex type of graph and precomputed_vertex_out_weight_sums must match",
                     *error);
    }
    if(initial_guess_vertices != nullptr)
    {
        CAPI_EXPECTS(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->vertex_type_
                         == reinterpret_cast<
                                rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                                initial_guess_vertices)
                                ->type_,
                     rocgraph_status_invalid_input,
                     "vertex type of graph and initial_guess_vertices must match",
                     *error);
        CAPI_EXPECTS(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->weight_type_
                         == reinterpret_cast<
                                rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                                initial_guess_values)
                                ->type_,
                     rocgraph_status_invalid_input,
                     "vertex type of graph and initial_guess_values must match",
                     *error);
    }
    if(personalization_vertices != nullptr)
    {
        CAPI_EXPECTS(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->vertex_type_
                         == reinterpret_cast<
                                rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                                personalization_vertices)
                                ->type_,
                     rocgraph_status_invalid_input,
                     "vertex type of graph and personalization_vector must match",
                     *error);
        CAPI_EXPECTS(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->weight_type_
                         == reinterpret_cast<
                                rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                                personalization_values)
                                ->type_,
                     rocgraph_status_invalid_input,
                     "vertex type of graph and personalization_vector must match",
                     *error);
    }

    pagerank_functor functor(handle,
                             graph,
                             precomputed_vertex_out_weight_vertices,
                             precomputed_vertex_out_weight_sums,
                             initial_guess_vertices,
                             initial_guess_values,
                             personalization_vertices,
                             personalization_values,
                             alpha,
                             epsilon,
                             max_iterations,
                             do_expensive_check);

    auto return_value = rocgraph::c_api::run_algorithm(graph, functor, result, error);

    CAPI_EXPECTS(rocgraph_centrality_result_converged(*result) == rocgraph_bool_true,
                 rocgraph_status_unknown_error,
                 "PageRank failed to converge.",
                 *error);

    return return_value;
}

extern "C" rocgraph_status rocgraph_personalized_pagerank_allow_nonconvergence(
    const rocgraph_handle_t*                        handle,
    rocgraph_graph_t*                               graph,
    const rocgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices,
    const rocgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
    const rocgraph_type_erased_device_array_view_t* initial_guess_vertices,
    const rocgraph_type_erased_device_array_view_t* initial_guess_values,
    const rocgraph_type_erased_device_array_view_t* personalization_vertices,
    const rocgraph_type_erased_device_array_view_t* personalization_values,
    double                                          alpha,
    double                                          epsilon,
    size_t                                          max_iterations,
    rocgraph_bool                                   do_expensive_check,
    rocgraph_centrality_result_t**                  result,
    rocgraph_error_t**                              error)
{
    if(precomputed_vertex_out_weight_vertices != nullptr)
    {
        CAPI_EXPECTS(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->vertex_type_
                         == reinterpret_cast<
                                rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                                precomputed_vertex_out_weight_vertices)
                                ->type_,
                     rocgraph_status_invalid_input,
                     "vertex type of graph and precomputed_vertex_out_weight_vertices must match",
                     *error);
        CAPI_EXPECTS(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->weight_type_
                         == reinterpret_cast<
                                rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                                precomputed_vertex_out_weight_sums)
                                ->type_,
                     rocgraph_status_invalid_input,
                     "vertex type of graph and precomputed_vertex_out_weight_sums must match",
                     *error);
    }
    if(initial_guess_vertices != nullptr)
    {
        CAPI_EXPECTS(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->vertex_type_
                         == reinterpret_cast<
                                rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                                initial_guess_vertices)
                                ->type_,
                     rocgraph_status_invalid_input,
                     "vertex type of graph and initial_guess_vertices must match",
                     *error);
        CAPI_EXPECTS(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->weight_type_
                         == reinterpret_cast<
                                rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                                initial_guess_values)
                                ->type_,
                     rocgraph_status_invalid_input,
                     "vertex type of graph and initial_guess_values must match",
                     *error);
    }
    if(personalization_vertices != nullptr)
    {
        CAPI_EXPECTS(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->vertex_type_
                         == reinterpret_cast<
                                rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                                personalization_vertices)
                                ->type_,
                     rocgraph_status_invalid_input,
                     "vertex type of graph and personalization_vector must match",
                     *error);
        CAPI_EXPECTS(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->weight_type_
                         == reinterpret_cast<
                                rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                                personalization_values)
                                ->type_,
                     rocgraph_status_invalid_input,
                     "vertex type of graph and personalization_vector must match",
                     *error);
    }

    pagerank_functor functor(handle,
                             graph,
                             precomputed_vertex_out_weight_vertices,
                             precomputed_vertex_out_weight_sums,
                             initial_guess_vertices,
                             initial_guess_values,
                             personalization_vertices,
                             personalization_values,
                             alpha,
                             epsilon,
                             max_iterations,
                             do_expensive_check);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}
