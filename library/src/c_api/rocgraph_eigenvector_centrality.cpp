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
#include "detail/utility_wrappers.hpp"
#include "graph_functions.hpp"

#include <optional>

namespace
{

    struct eigenvector_centrality_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                          handle_;
        rocgraph::c_api::rocgraph_graph_t*             graph_{};
        double                                         epsilon_{};
        size_t                                         max_iterations_{};
        bool                                           do_expensive_check_{};
        rocgraph::c_api::rocgraph_centrality_result_t* result_{};

        eigenvector_centrality_functor(rocgraph_handle_t const* handle,
                                       rocgraph_graph_t*        graph,
                                       double                   epsilon,
                                       size_t                   max_iterations,
                                       bool                     do_expensive_check)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
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
                // Eigenvector Centrality expects store_transposed == true
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

                auto centralities
                    = rocgraph::eigenvector_centrality<vertex_t, edge_t, weight_t, multi_gpu>(
                        handle_,
                        graph_view,
                        (edge_weights != nullptr) ? std::make_optional(edge_weights->view())
                                                  : std::nullopt,
                        std::optional<raft::device_span<weight_t>>{},
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
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(centralities,
                                                                             graph_->weight_type_)};
            }
        }
    };

} // namespace

extern "C" rocgraph_status rocgraph_eigenvector_centrality(const rocgraph_handle_t* handle,
                                                           rocgraph_graph_t*        graph,
                                                           double                   epsilon,
                                                           size_t                   max_iterations,
                                                           rocgraph_bool do_expensive_check,
                                                           rocgraph_centrality_result_t** result,
                                                           rocgraph_error_t**             error)
{
    eigenvector_centrality_functor functor(
        handle, graph, epsilon, max_iterations, do_expensive_check);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}
