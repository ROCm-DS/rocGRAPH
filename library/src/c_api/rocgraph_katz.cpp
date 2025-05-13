// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_abstract_functor.hpp"
#include "c_api/rocgraph_centrality_result.hpp"
#include "c_api/rocgraph_graph.hpp"
#include "c_api/rocgraph_handle.hpp"
#include "c_api/rocgraph_utils.hpp"

#include "internal/rocgraph_algorithms.h"

#include "algorithms.hpp"
#include "detail/shuffle_wrappers.hpp"
#include "detail/utility_wrappers.hpp"
#include "graph_functions.hpp"

#include <optional>

namespace
{

    struct katz_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                                            handle_;
        rocgraph::c_api::rocgraph_graph_t*                               graph_{};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* betas_{};
        double                                                           alpha_{};
        double                                                           beta_{};
        double                                                           epsilon_{};
        size_t                                                           max_iterations_{};
        bool                                                             do_expensive_check_{};
        rocgraph::c_api::rocgraph_centrality_result_t*                   result_{};

        katz_functor(rocgraph_handle_t const*                        handle,
                     rocgraph_graph_t*                               graph,
                     rocgraph_type_erased_device_array_view_t const* betas,
                     double                                          alpha,
                     double                                          beta,
                     double                                          epsilon,
                     size_t                                          max_iterations,
                     bool                                            do_expensive_check)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , betas_(reinterpret_cast<
                     rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(betas))
            , alpha_(alpha)
            , beta_(beta)
            , epsilon_(epsilon)
            , max_iterations_(max_iterations)
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
            if constexpr(!rocgraph::is_candidate<vertex_t, edge_t, weight_t>::value)
            {
                unsupported();
            }
            else
            {
                // katz expects store_transposed == true
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

                rmm::device_uvector<weight_t> centralities(
                    graph_view.local_vertex_partition_range_size(), handle_.get_stream());
                rmm::device_uvector<weight_t> betas(0, handle_.get_stream());

                if(betas_ != nullptr)
                {
                    rmm::device_uvector<vertex_t> betas_vertex_ids(
                        graph_view.local_vertex_partition_range_size(), handle_.get_stream());
                    rocgraph::detail::sequence_fill(
                        handle_.get_stream(),
                        betas_vertex_ids.data(),
                        betas_vertex_ids.size(),
                        graph_view.local_vertex_partition_range_first());

                    betas.resize(graph_view.local_vertex_partition_range_size(),
                                 handle_.get_stream());

                    raft::copy(betas.data(),
                               betas_->as_type<weight_t>(),
                               betas.size(),
                               handle_.get_stream());

                    betas
                        = rocgraph::detail::collect_local_vertex_values_from_ext_vertex_value_pairs<
                            vertex_t,
                            weight_t,
                            multi_gpu>(handle_,
                                       std::move(betas_vertex_ids),
                                       std::move(betas),
                                       *number_map,
                                       graph_view.local_vertex_partition_range_first(),
                                       graph_view.local_vertex_partition_range_last(),
                                       weight_t{0},
                                       do_expensive_check_);
                }

                rocgraph::katz_centrality<vertex_t, edge_t, weight_t, weight_t, multi_gpu>(
                    handle_,
                    graph_view,
                    (edge_weights != nullptr) ? std::make_optional(edge_weights->view())
                                              : std::nullopt,
                    betas_ == nullptr ? nullptr : betas.data(),
                    centralities.data(),
                    static_cast<weight_t>(alpha_),
                    static_cast<weight_t>(beta_),
                    static_cast<weight_t>(epsilon_),
                    max_iterations_,
                    false,
                    true,
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

} // namespace

extern "C" rocgraph_status
    rocgraph_katz_centrality(const rocgraph_handle_t*                        handle,
                             rocgraph_graph_t*                               graph,
                             const rocgraph_type_erased_device_array_view_t* betas,
                             double                                          alpha,
                             double                                          beta,
                             double                                          epsilon,
                             size_t                                          max_iterations,
                             rocgraph_bool                                   do_expensive_check,
                             rocgraph_centrality_result_t**                  result,
                             rocgraph_error_t**                              error)
{
    katz_functor functor(
        handle, graph, nullptr, alpha, beta, epsilon, max_iterations, do_expensive_check);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}
