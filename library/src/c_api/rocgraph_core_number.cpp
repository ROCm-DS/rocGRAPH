// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_abstract_functor.hpp"
#include "c_api/rocgraph_core_result.hpp"
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

    struct core_number_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                    handle_;
        rocgraph::c_api::rocgraph_graph_t*       graph_{};
        rocgraph::k_core_degree_type_t           degree_type_{};
        bool                                     do_expensive_check_{};
        rocgraph::c_api::rocgraph_core_result_t* result_{};

        core_number_functor(rocgraph_handle_t const*    handle,
                            rocgraph_graph_t*           graph,
                            rocgraph_k_core_degree_type degree_type,
                            bool                        do_expensive_check)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , degree_type_(static_cast<rocgraph::k_core_degree_type_t>(degree_type))
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

                rmm::device_uvector<edge_t> core_numbers(
                    graph_view.local_vertex_partition_range_size(), handle_.get_stream());

                auto degree_type = reinterpret_cast<rocgraph::k_core_degree_type_t>(degree_type_);

                rocgraph::core_number<vertex_t, edge_t, multi_gpu>(
                    handle_,
                    graph_view,
                    core_numbers.data(),
                    degree_type,
                    size_t{0},
                    std::numeric_limits<size_t>::max(),
                    do_expensive_check_);

                rmm::device_uvector<vertex_t> vertex_ids(
                    graph_view.local_vertex_partition_range_size(), handle_.get_stream());
                raft::copy(
                    vertex_ids.data(), number_map->data(), vertex_ids.size(), handle_.get_stream());

                result_ = new rocgraph::c_api::rocgraph_core_result_t{
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(vertex_ids,
                                                                             graph_->vertex_type_),
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(core_numbers,
                                                                             graph_->edge_type_)};
            }
        }
    };

} // namespace

extern "C" rocgraph_status rocgraph_core_number(const rocgraph_handle_t*    handle,
                                                rocgraph_graph_t*           graph,
                                                rocgraph_k_core_degree_type degree_type,
                                                rocgraph_bool               do_expensive_check,
                                                rocgraph_core_result_t**    result,
                                                rocgraph_error_t**          error)
{
    core_number_functor functor(handle, graph, degree_type, do_expensive_check);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}
