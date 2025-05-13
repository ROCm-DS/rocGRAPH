// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "internal/rocgraph_community_algorithms.h"

#include "algorithms.hpp"
#include "detail/shuffle_wrappers.hpp"
#include "detail/utility_wrappers.hpp"
#include "graph_functions.hpp"

#include "c_api/rocgraph_abstract_functor.hpp"
#include "c_api/rocgraph_graph.hpp"
#include "c_api/rocgraph_graph_helper.hpp"
#include "c_api/rocgraph_handle.hpp"
#include "c_api/rocgraph_hierarchical_clustering_result.hpp"
#include "c_api/rocgraph_random.hpp"
#include "c_api/rocgraph_utils.hpp"

#include <optional>

namespace
{

    struct ecg_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                                       handle_;
        rocgraph::c_api::rocgraph_rng_state_t*                      rng_state_{nullptr};
        rocgraph::c_api::rocgraph_graph_t*                          graph_{nullptr};
        double                                                      min_weight_{0.1};
        size_t                                                      ensemble_size_{10};
        size_t                                                      max_level_{0};
        double                                                      threshold_{0.001};
        double                                                      resolution_{1};
        bool                                                        do_expensive_check_{false};
        rocgraph::c_api::rocgraph_hierarchical_clustering_result_t* result_{};

        ecg_functor(::rocgraph_handle_t const* handle,
                    ::rocgraph_rng_state_t*    rng_state,
                    ::rocgraph_graph_t*        graph,
                    double                     min_weight,
                    size_t                     ensemble_size,
                    size_t                     max_level,
                    double                     threshold,
                    double                     resolution,
                    bool                       do_expensive_check)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , rng_state_(reinterpret_cast<rocgraph::c_api::rocgraph_rng_state_t*>(rng_state))
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , max_level_(max_level)
            , threshold_(threshold)
            , resolution_(resolution)
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
                // ecg expects store_transposed == false
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

                rmm::device_uvector<vertex_t> clusters(0, handle_.get_stream());

                weight_t modularity;

                std::tie(clusters, std::ignore, modularity) = rocgraph::ecg(
                    handle_,
                    rng_state_->rng_state_,
                    graph_view,
                    (edge_weights != nullptr)
                        ? std::make_optional(edge_weights->view())
                        : std::make_optional(rocgraph::c_api::create_constant_edge_property(
                                                 handle_, graph_view, weight_t{1})
                                                 .view()),
                    static_cast<weight_t>(min_weight_),
                    ensemble_size_,
                    max_level_,
                    static_cast<weight_t>(threshold_),
                    static_cast<weight_t>(resolution_));

                rmm::device_uvector<vertex_t> vertices(
                    graph_view.local_vertex_partition_range_size(), handle_.get_stream());
                raft::copy(
                    vertices.data(), number_map->data(), vertices.size(), handle_.get_stream());

                result_ = new rocgraph::c_api::rocgraph_hierarchical_clustering_result_t{
                    modularity,
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(vertices,
                                                                             graph_->vertex_type_),
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(clusters,
                                                                             graph_->vertex_type_)};
            }
        }
    };

} // namespace

extern "C" rocgraph_status rocgraph_ecg(const rocgraph_handle_t* handle,
                                        rocgraph_rng_state_t*    rng_state,
                                        rocgraph_graph_t*        graph,
                                        double                   min_weight,
                                        size_t                   ensemble_size,
                                        size_t                   max_level,
                                        double                   threshold,
                                        double                   resolution,
                                        rocgraph_bool            do_expensive_check,
                                        rocgraph_hierarchical_clustering_result_t** result,
                                        rocgraph_error_t**                          error)
{
    ecg_functor functor(handle,
                        rng_state,
                        graph,
                        min_weight,
                        ensemble_size,
                        max_level,
                        threshold,
                        resolution,
                        do_expensive_check);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}
