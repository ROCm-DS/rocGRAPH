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

#include "algorithms.hpp"
#include "detail/shuffle_wrappers.hpp"
#include "detail/utility_wrappers.hpp"
#include "graph_functions.hpp"
#include "utilities/device_comm.hpp"
#include "utilities/host_scalar_comm.hpp"

#include <numeric>
#include <optional>

namespace
{

    struct extract_ego_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                                            handle_;
        rocgraph::c_api::rocgraph_graph_t*                               graph_;
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* source_vertices_;
        size_t                                                           radius_;
        bool                                                             do_expensive_check_;
        rocgraph::c_api::rocgraph_induced_subgraph_result_t*             result_{};

        extract_ego_functor(::rocgraph_handle_t const*                        handle,
                            ::rocgraph_graph_t*                               graph,
                            ::rocgraph_type_erased_device_array_view_t const* source_vertices,
                            size_t                                            radius,
                            bool                                              do_expensive_check)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , source_vertices_(reinterpret_cast<
                               rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                  source_vertices))
            , radius_(radius)
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
                // extract ego expects store_transposed == false
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

                rmm::device_uvector<vertex_t> source_vertices(source_vertices_->size_,
                                                              handle_.get_stream());
                raft::copy(source_vertices.data(),
                           source_vertices_->as_type<vertex_t>(),
                           source_vertices.size(),
                           handle_.get_stream());

                std::optional<rmm::device_uvector<size_t>> source_indices{std::nullopt};

                if constexpr(multi_gpu)
                {
                    auto displacements = rocgraph::host_scalar_allgather(
                        handle_.get_comms(), source_vertices.size(), handle_.get_stream());
                    std::exclusive_scan(displacements.begin(),
                                        displacements.end(),
                                        displacements.begin(),
                                        size_t{0});
                    source_indices
                        = rmm::device_uvector<size_t>(source_vertices.size(), handle_.get_stream());
                    rocgraph::detail::sequence_fill(handle_.get_stream(),
                                                    (*source_indices).data(),
                                                    (*source_indices).size(),
                                                    displacements[handle_.get_comms().get_rank()]);

                    std::tie(source_vertices, source_indices) = rocgraph::detail::
                        shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
                            handle_, std::move(source_vertices), std::move(*source_indices));
                }

                rocgraph::renumber_ext_vertices<vertex_t, multi_gpu>(
                    handle_,
                    source_vertices.data(),
                    source_vertices.size(),
                    number_map->data(),
                    graph_view.local_vertex_partition_range_first(),
                    graph_view.local_vertex_partition_range_last(),
                    do_expensive_check_);

                auto [src, dst, wgt, edge_offsets]
                    = rocgraph::extract_ego<vertex_t, edge_t, weight_t, multi_gpu>(
                        handle_,
                        graph_view,
                        (edge_weights != nullptr) ? std::make_optional(edge_weights->view())
                                                  : std::nullopt,
                        raft::device_span<vertex_t const>{source_vertices.data(),
                                                          source_vertices.size()},
                        radius_,
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

                if constexpr(multi_gpu)
                {
                    auto recvcounts = rocgraph::host_scalar_allgather(
                        handle_.get_comms(), (*source_indices).size(), handle_.get_stream());
                    std::vector<size_t> displacements(recvcounts.size());
                    std::exclusive_scan(
                        recvcounts.begin(), recvcounts.end(), displacements.begin(), size_t{0});
                    rmm::device_uvector<size_t> allgathered_indices(
                        displacements.back() + recvcounts.back(), handle_.get_stream());
                    rocgraph::device_allgatherv(handle_.get_comms(),
                                                (*source_indices).begin(),
                                                allgathered_indices.begin(),
                                                recvcounts,
                                                displacements,
                                                handle_.get_stream());
                    source_indices = std::move(allgathered_indices);

                    std::tie(edge_offsets, src, dst, wgt)
                        = rocgraph::c_api::detail::reorder_extracted_egonets<vertex_t, weight_t>(
                            handle_,
                            std::move(*source_indices),
                            std::move(edge_offsets),
                            std::move(src),
                            std::move(dst),
                            std::move(wgt));
                }

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
                        edge_offsets, rocgraph_data_type_id_size_t)};
            }
        }
    };

} // namespace

extern "C" rocgraph_status
    rocgraph_extract_ego(const rocgraph_handle_t*                        handle,
                         rocgraph_graph_t*                               graph,
                         const rocgraph_type_erased_device_array_view_t* source_vertices,
                         size_t                                          radius,
                         rocgraph_bool                                   do_expensive_check,
                         rocgraph_induced_subgraph_result_t**            result,
                         rocgraph_error_t**                              error)
{
    extract_ego_functor functor(handle, graph, source_vertices, radius, do_expensive_check);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}
