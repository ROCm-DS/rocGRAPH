// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_abstract_functor.hpp"
#include "c_api/rocgraph_capi_helper.hpp"
#include "c_api/rocgraph_graph.hpp"
#include "c_api/rocgraph_handle.hpp"
#include "c_api/rocgraph_induced_subgraph_result.hpp"
#include "c_api/rocgraph_utils.hpp"

#include "internal/rocgraph_algorithms.h"
#include "internal/rocgraph_graph_functions.h"

#include "algorithms.hpp"
#include "detail/shuffle_wrappers.hpp"
#include "detail/utility_wrappers.hpp"
#include "graph_functions.hpp"

namespace
{

    struct induced_subgraph_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                                            handle_;
        rocgraph::c_api::rocgraph_graph_t*                               graph_{};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* subgraph_offsets_{};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* subgraph_vertices_{};
        bool                                                             do_expensive_check_{};
        rocgraph::c_api::rocgraph_induced_subgraph_result_t*             result_{};

        induced_subgraph_functor(rocgraph_handle_t const*                        handle,
                                 rocgraph_graph_t*                               graph,
                                 rocgraph_type_erased_device_array_view_t const* subgraph_offsets,
                                 rocgraph_type_erased_device_array_view_t const* subgraph_vertices,
                                 bool                                            do_expensive_check)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , subgraph_offsets_(reinterpret_cast<
                                rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                  subgraph_offsets))
            , subgraph_vertices_(reinterpret_cast<
                                 rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                  subgraph_vertices))
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
                // induced subgraph expects store_transposed == false
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

                rmm::device_uvector<size_t>   subgraph_offsets(0, handle_.get_stream());
                rmm::device_uvector<vertex_t> subgraph_vertices(subgraph_vertices_->size_,
                                                                handle_.get_stream());

                raft::copy(subgraph_vertices.data(),
                           subgraph_vertices_->as_type<vertex_t>(),
                           subgraph_vertices_->size_,
                           handle_.get_stream());

                if constexpr(multi_gpu)
                {
                    std::tie(subgraph_vertices, subgraph_offsets)
                        = rocgraph::c_api::detail::shuffle_vertex_ids_and_offsets(
                            handle_,
                            std::move(subgraph_vertices),
                            raft::device_span<size_t const>{subgraph_offsets_->as_type<size_t>(),
                                                            subgraph_offsets_->size_});
                }
                else
                {
                    subgraph_offsets.resize(subgraph_offsets_->size_, handle_.get_stream());
                    raft::copy(subgraph_offsets.data(),
                               subgraph_offsets_->as_type<size_t>(),
                               subgraph_offsets_->size_,
                               handle_.get_stream());
                }

                //
                // Need to renumber subgraph_vertices
                //
                rocgraph::renumber_local_ext_vertices<vertex_t, multi_gpu>(
                    handle_,
                    subgraph_vertices.data(),
                    subgraph_vertices.size(),
                    number_map->data(),
                    graph_view.local_vertex_partition_range_first(),
                    graph_view.local_vertex_partition_range_last(),
                    do_expensive_check_);

                auto [src, dst, wgt, graph_offsets] = rocgraph::extract_induced_subgraphs(
                    handle_,
                    graph_view,
                    (edge_weights != nullptr) ? std::make_optional(edge_weights->view())
                                              : std::nullopt,
                    raft::device_span<size_t const>{subgraph_offsets.data(),
                                                    subgraph_offsets.size()},
                    raft::device_span<vertex_t const>{subgraph_vertices.data(),
                                                      subgraph_vertices.size()},
                    do_expensive_check_);

                rocgraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
                    handle_,
                    src.data(),
                    src.size(),
                    number_map->data(),
                    graph_view.vertex_partition_range_lasts(),
                    do_expensive_check_);

                rocgraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
                    handle_,
                    dst.data(),
                    dst.size(),
                    number_map->data(),
                    graph_view.vertex_partition_range_lasts(),
                    do_expensive_check_);

                // FIXME: Add support for edge_id and edge_type_id.
                result_ = new rocgraph::c_api::rocgraph_induced_subgraph_result_t{
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(src,
                                                                             graph_->vertex_type_),
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(dst,
                                                                             graph_->vertex_type_),
                    wgt ? new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                              *wgt, graph_->weight_type_)
                        : NULL,
                    NULL,
                    NULL,
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                        graph_offsets, rocgraph_data_type_id_size_t)};
            }
        }
    };

} // namespace

extern "C" rocgraph_status rocgraph_extract_induced_subgraph(
    const rocgraph_handle_t*                        handle,
    rocgraph_graph_t*                               graph,
    const rocgraph_type_erased_device_array_view_t* subgraph_offsets,
    const rocgraph_type_erased_device_array_view_t* subgraph_vertices,
    rocgraph_bool                                   do_expensive_check,
    rocgraph_induced_subgraph_result_t**            result,
    rocgraph_error_t**                              error)
{
    CAPI_EXPECTS(
        reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->vertex_type_
            == reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                   subgraph_vertices)
                   ->type_,
        rocgraph_status_invalid_input,
        "vertex type of graph and subgraph_vertices must match",
        *error);

    induced_subgraph_functor functor(
        handle, graph, subgraph_offsets, subgraph_vertices, do_expensive_check);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}
