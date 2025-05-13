// Copyright (C) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_abstract_functor.hpp"
#include "c_api/rocgraph_graph.hpp"
#include "c_api/rocgraph_handle.hpp"
#include "c_api/rocgraph_hierarchical_clustering_result.hpp"
#include "c_api/rocgraph_utils.hpp"

#include "internal/rocgraph_algorithms.h"

#include "algorithms.hpp"
#include "detail/shuffle_wrappers.hpp"
#include "detail/utility_wrappers.hpp"
#include "graph_functions.hpp"

#include <optional>

namespace
{

    struct legacy_ecg_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                                       handle_;
        rocgraph::c_api::rocgraph_graph_t*                          graph_;
        double                                                      min_weight_;
        size_t                                                      ensemble_size_;
        bool                                                        do_expensive_check_;
        rocgraph::c_api::rocgraph_hierarchical_clustering_result_t* result_{};

        legacy_ecg_functor(::rocgraph_handle_t const* handle,
                           ::rocgraph_graph_t*        graph,
                           double                     min_weight,
                           size_t                     ensemble_size,
                           bool                       do_expensive_check)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , min_weight_(min_weight)
            , ensemble_size_(ensemble_size)
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
            else if constexpr(multi_gpu)
            {
                unsupported();
            }
            else if constexpr(!std::is_same_v<edge_t, int32_t>)
            {
                unsupported();
            }
            else
            {
                // ecg expects store_transposed == false
                if constexpr(store_transposed)
                {
                    status_ = rocgraph::c_api::
                        transpose_storage<vertex_t, edge_t, weight_t, store_transposed, false>(
                            handle_, graph_, error_.get());
                    if(status_ != rocgraph_status_success)
                        return;
                }

                auto graph = reinterpret_cast<rocgraph::graph_t<vertex_t, edge_t, false, false>*>(
                    graph_->graph_);

                auto edge_weights = reinterpret_cast<rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, false, false>,
                    weight_t>*>(graph_->edge_weights_);

                auto number_map
                    = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

                auto graph_view = graph->view();

                auto edge_partition_view = graph_view.local_edge_partition_view();

                rocgraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> legacy_graph_view(
                    const_cast<edge_t*>(edge_partition_view.offsets().data()),
                    const_cast<vertex_t*>(edge_partition_view.indices().data()),
                    const_cast<weight_t*>(edge_weights->view().value_firsts().front()),
                    edge_partition_view.offsets().size() - 1,
                    edge_partition_view.indices().size());

                rmm::device_uvector<vertex_t> clusters(
                    graph_view.local_vertex_partition_range_size(), handle_.get_stream());

                // FIXME:  Need modularity..., although currently not used
                rocgraph::ecg(handle_,
                              legacy_graph_view,
                              static_cast<weight_t>(min_weight_),
                              static_cast<vertex_t>(ensemble_size_),
                              clusters.data());

                rmm::device_uvector<vertex_t> vertices(
                    graph_view.local_vertex_partition_range_size(), handle_.get_stream());
                raft::copy(
                    vertices.data(), number_map->data(), vertices.size(), handle_.get_stream());

                result_ = new rocgraph::c_api::rocgraph_hierarchical_clustering_result_t{
                    weight_t{0},
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(vertices,
                                                                             graph_->vertex_type_),
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(clusters,
                                                                             graph_->vertex_type_)};
            }
        }
    };

} // namespace

extern "C" rocgraph_status rocgraph_legacy_ecg(const rocgraph_handle_t* handle,
                                               rocgraph_graph_t*        graph,
                                               double                   min_weight,
                                               size_t                   ensemble_size,
                                               rocgraph_bool            do_expensive_check,
                                               rocgraph_hierarchical_clustering_result_t** result,
                                               rocgraph_error_t**                          error)
{
    legacy_ecg_functor functor(handle, graph, min_weight, ensemble_size, do_expensive_check);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}
