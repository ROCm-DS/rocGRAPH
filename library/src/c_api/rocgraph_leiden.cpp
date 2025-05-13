// Copyright (C) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_abstract_functor.hpp"
#include "c_api/rocgraph_graph.hpp"
#include "c_api/rocgraph_graph_helper.hpp"
#include "c_api/rocgraph_handle.hpp"
#include "c_api/rocgraph_hierarchical_clustering_result.hpp"
#include "c_api/rocgraph_random.hpp"
#include "c_api/rocgraph_utils.hpp"

#include "internal/rocgraph_community_algorithms.h"

#include "algorithms.hpp"
#include "detail/shuffle_wrappers.hpp"
#include "detail/utility_wrappers.hpp"
#include "graph_functions.hpp"

#include <raft/core/handle.hpp>

#include <optional>

namespace
{

    struct leiden_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                                       handle_;
        rocgraph::c_api::rocgraph_rng_state_t*                      rng_state_{nullptr};
        rocgraph::c_api::rocgraph_graph_t*                          graph_{nullptr};
        size_t                                                      max_level_;
        double                                                      resolution_;
        bool                                                        do_expensive_check_;
        rocgraph::c_api::rocgraph_hierarchical_clustering_result_t* result_{};

        leiden_functor(::rocgraph_handle_t const* handle,
                       rocgraph_rng_state_t*      rng_state,
                       ::rocgraph_graph_t*        graph,
                       size_t                     max_level,
                       double                     resolution,
                       bool                       do_expensive_check)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , rng_state_(reinterpret_cast<rocgraph::c_api::rocgraph_rng_state_t*>(rng_state))
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , max_level_(max_level)
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
                // leiden expects store_transposed == false
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

                rmm::device_uvector<vertex_t> clusters(
                    graph_view.local_vertex_partition_range_size(), handle_.get_stream());

                // FIXME: Revisit the constant edge property idea.  We could consider an alternate
                // implementation (perhaps involving the thrust::constant_iterator), or we
                // could add support in Leiden for std::nullopt as the edge weights behaving
                // as desired and only instantiating a real edge_property_view_t for the
                // coarsened graphs.
                auto [level, modularity] = rocgraph::leiden(
                    handle_,
                    rng_state_->rng_state_,
                    graph_view,
                    (edge_weights != nullptr)
                        ? std::make_optional(edge_weights->view())
                        : std::make_optional(rocgraph::c_api::create_constant_edge_property(
                                                 handle_, graph_view, weight_t{1})
                                                 .view()),
                    clusters.data(),
                    max_level_,
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

extern "C" rocgraph_status rocgraph_leiden(const rocgraph_handle_t* handle,
                                           rocgraph_rng_state_t*    rng_state,
                                           rocgraph_graph_t*        graph,
                                           size_t                   max_level,
                                           double                   resolution,
                                           double                   theta,
                                           rocgraph_bool            do_expensive_check,
                                           rocgraph_hierarchical_clustering_result_t** result,
                                           rocgraph_error_t**                          error)
{
    leiden_functor functor(handle, rng_state, graph, max_level, resolution, do_expensive_check);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}
