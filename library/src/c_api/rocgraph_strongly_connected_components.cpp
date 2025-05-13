// Copyright (C) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_abstract_functor.hpp"
#include "c_api/rocgraph_graph.hpp"
#include "c_api/rocgraph_handle.hpp"
#include "c_api/rocgraph_labeling_result.hpp"
#include "c_api/rocgraph_utils.hpp"

#include "internal/rocgraph_labeling_algorithms.h"

#include "algorithms.hpp"
#include "detail/shuffle_wrappers.hpp"
#include "detail/utility_wrappers.hpp"
#include "graph_functions.hpp"

#include <optional>

namespace
{

    struct scc_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                        handle_;
        rocgraph::c_api::rocgraph_graph_t*           graph_{};
        bool                                         do_expensive_check_{};
        rocgraph::c_api::rocgraph_labeling_result_t* result_{};

        scc_functor(::rocgraph_handle_t const* handle,
                    ::rocgraph_graph_t*        graph,
                    bool                       do_expensive_check)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
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
                status_ = rocgraph_status_not_implemented;
                error_->error_message_
                    = "strongly connected components not currently implemented for multi-GPU";
            }
            else if constexpr(!std::is_same_v<vertex_t, edge_t>)
            {
                unsupported();
            }
            else if constexpr(std::is_same_v<weight_t, double>)
            {
                unsupported();
            }
            else
            {
                // SCC expects store_transposed == false
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

                auto number_map
                    = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

                auto graph_view          = graph->view();
                auto edge_partition_view = graph_view.local_edge_partition_view();

                rocgraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> legacy_graph_view(
                    const_cast<edge_t*>(edge_partition_view.offsets().data()),
                    const_cast<vertex_t*>(edge_partition_view.indices().data()),
                    nullptr,
                    edge_partition_view.offsets().size() - 1,
                    edge_partition_view.indices().size());

                rmm::device_uvector<vertex_t> components(graph_view.number_of_vertices(),
                                                         handle_.get_stream());

                rocgraph::connected_components(
                    legacy_graph_view, rocgraph::rocgraph_cc_t::ROCGRAPH_STRONG, components.data());
                rmm::device_uvector<vertex_t> vertex_ids(
                    graph_view.local_vertex_partition_range_size(), handle_.get_stream());
                raft::copy(
                    vertex_ids.data(), number_map->data(), vertex_ids.size(), handle_.get_stream());

                result_ = new rocgraph::c_api::rocgraph_labeling_result_t{
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(vertex_ids,
                                                                             graph_->vertex_type_),
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(components,
                                                                             graph_->vertex_type_)};
            }
        }
    };

} // namespace

extern "C" rocgraph_status
    rocgraph_strongly_connected_components(const rocgraph_handle_t*     handle,
                                           rocgraph_graph_t*            graph,
                                           rocgraph_bool                do_expensive_check,
                                           rocgraph_labeling_result_t** result,
                                           rocgraph_error_t**           error)
{
    scc_functor functor(handle, graph, do_expensive_check);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}
