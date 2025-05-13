// Copyright (C) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_abstract_functor.hpp"
#include "c_api/rocgraph_graph.hpp"
#include "c_api/rocgraph_handle.hpp"
#include "c_api/rocgraph_induced_subgraph_result.hpp"
#include "c_api/rocgraph_utils.hpp"

#include "internal/rocgraph_algorithms.h"

#include "algorithms.hpp"
#include "detail/shuffle_wrappers.hpp"
#include "detail/utility_wrappers.hpp"
#include "graph_functions.hpp"

#include <optional>

namespace
{

    struct k_truss_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                                handle_;
        rocgraph::c_api::rocgraph_graph_t*                   graph_;
        size_t                                               k_;
        bool                                                 do_expensive_check_;
        rocgraph::c_api::rocgraph_induced_subgraph_result_t* result_{};

        k_truss_functor(::rocgraph_handle_t const* handle,
                        ::rocgraph_graph_t*        graph,
                        size_t                     k,
                        bool                       do_expensive_check)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , k_(k)
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
            else
            {
                // k_truss expects store_transposed == false
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

                auto edge_weights = reinterpret_cast<rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                    weight_t>*>(graph_->edge_weights_);

                auto number_map
                    = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

                auto graph_view = graph->view();

                auto [result_src, result_dst, result_wgt]
                    = rocgraph::k_truss<vertex_t, edge_t, weight_t, multi_gpu>(
                        handle_,
                        graph_view,
                        edge_weights ? std::make_optional(edge_weights->view()) : std::nullopt,
                        k_,
                        do_expensive_check_);

                rocgraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
                    handle_,
                    result_src.data(),
                    result_src.size(),
                    number_map->data(),
                    graph_view.vertex_partition_range_lasts(),
                    do_expensive_check_);

                rocgraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
                    handle_,
                    result_dst.data(),
                    result_dst.size(),
                    number_map->data(),
                    graph_view.vertex_partition_range_lasts(),
                    do_expensive_check_);

                rmm::device_uvector<size_t> edge_offsets(2, handle_.get_stream());
                std::vector<size_t>         h_edge_offsets{{0, result_src.size()}};
                raft::update_device(edge_offsets.data(),
                                    h_edge_offsets.data(),
                                    h_edge_offsets.size(),
                                    handle_.get_stream());

                // FIXME: Add support for edge_id and edge_type_id.
                result_ = new rocgraph::c_api::rocgraph_induced_subgraph_result_t{
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(result_src,
                                                                             graph_->vertex_type_),
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(result_dst,
                                                                             graph_->vertex_type_),
                    result_wgt ? new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                                     *result_wgt, graph_->weight_type_)
                               : NULL,
                    NULL,
                    NULL,
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                        edge_offsets, rocgraph_data_type_id_size_t)};
            }
        }
    };

} // namespace

extern "C" rocgraph_status rocgraph_k_truss_subgraph(const rocgraph_handle_t* handle,
                                                     rocgraph_graph_t*        graph,
                                                     size_t                   k,
                                                     rocgraph_bool            do_expensive_check,
                                                     rocgraph_induced_subgraph_result_t** result,
                                                     rocgraph_error_t**                   error)
{
    k_truss_functor functor(handle, graph, k, do_expensive_check);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}
