// Copyright (C) 2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_abstract_functor.hpp"
#include "c_api/rocgraph_degrees_result.hpp"
#include "c_api/rocgraph_graph.hpp"
#include "c_api/rocgraph_handle.hpp"
#include "c_api/rocgraph_utils.hpp"

#include "internal/rocgraph_algorithms.h"
#include "internal/rocgraph_graph_functions.h"

#include "algorithms.hpp"
#include "detail/shuffle_wrappers.hpp"
#include "detail/utility_wrappers.hpp"
#include "graph_functions.hpp"
#include "vertex_partition_device_view_device.hpp"

#include <thrust/gather.h>

#include <optional>

namespace
{

    struct degrees_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                                            handle_;
        rocgraph::c_api::rocgraph_graph_t*                               graph_{};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* source_vertices_;
        bool                                                             in_degrees_{false};
        bool                                                             out_degrees_{false};
        bool                                                             do_expensive_check_{false};
        rocgraph::c_api::rocgraph_degrees_result_t*                      result_{};

        degrees_functor(rocgraph_handle_t const*                          handle,
                        rocgraph_graph_t*                                 graph,
                        ::rocgraph_type_erased_device_array_view_t const* source_vertices,
                        bool                                              in_degrees,
                        bool                                              out_degrees,
                        bool                                              do_expensive_check)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , source_vertices_(reinterpret_cast<
                               rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                  source_vertices))
            , in_degrees_{in_degrees}
            , out_degrees_{out_degrees}
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
                auto graph = reinterpret_cast<
                    rocgraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>*>(
                    graph_->graph_);

                auto graph_view = graph->view();

                auto number_map
                    = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

                std::optional<rmm::device_uvector<edge_t>> in_degrees{std::nullopt};
                std::optional<rmm::device_uvector<edge_t>> out_degrees{std::nullopt};

                if(in_degrees_ && out_degrees_ && graph_view.is_symmetric())
                {
                    in_degrees = store_transposed ? graph_view.compute_in_degrees(handle_)
                                                  : graph_view.compute_out_degrees(handle_);
                    // out_degrees will be extracted from in_degrees in the result
                }
                else
                {
                    if(in_degrees_)
                        in_degrees = graph_view.compute_in_degrees(handle_);

                    if(out_degrees_)
                        out_degrees = graph_view.compute_out_degrees(handle_);
                }

                rmm::device_uvector<vertex_t> vertex_ids(0, handle_.get_stream());

                if(source_vertices_)
                {
                    // FIXME: Would be more efficient if graph_view.compute_*_degrees could take a vertex
                    //  subset
                    vertex_ids.resize(source_vertices_->size_, handle_.get_stream());
                    raft::copy(vertex_ids.data(),
                               source_vertices_->as_type<vertex_t>(),
                               vertex_ids.size(),
                               handle_.get_stream());

                    if constexpr(multi_gpu)
                    {
                        vertex_ids = rocgraph::detail::
                            shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
                                handle_, std::move(vertex_ids));
                    }

                    rocgraph::renumber_ext_vertices<vertex_t, multi_gpu>(
                        handle_,
                        vertex_ids.data(),
                        vertex_ids.size(),
                        number_map->data(),
                        graph_view.local_vertex_partition_range_first(),
                        graph_view.local_vertex_partition_range_last(),
                        do_expensive_check_);

                    auto vertex_partition
                        = rocgraph::vertex_partition_device_view_t<vertex_t, multi_gpu>(
                            graph_view.local_vertex_partition_view());

                    auto vertices_iter = thrust::make_transform_iterator(
                        vertex_ids.begin(), [vertex_partition] __device__(auto v) -> vertex_t {
                            return vertex_partition
                                .local_vertex_partition_offset_from_vertex_nocheck(v);
                        });

                    if(in_degrees && out_degrees)
                    {
                        rmm::device_uvector<edge_t> tmp_in_degrees(vertex_ids.size(),
                                                                   handle_.get_stream());
                        rmm::device_uvector<edge_t> tmp_out_degrees(vertex_ids.size(),
                                                                    handle_.get_stream());
                        thrust::gather(
                            handle_.get_thrust_policy(),
                            vertices_iter,
                            vertices_iter + vertex_ids.size(),
                            thrust::make_zip_iterator(in_degrees->begin(), out_degrees->begin()),
                            thrust::make_zip_iterator(tmp_in_degrees.begin(),
                                                      tmp_out_degrees.begin()));
                        in_degrees  = std::move(tmp_in_degrees);
                        out_degrees = std::move(tmp_out_degrees);
                    }
                    else if(in_degrees)
                    {
                        rmm::device_uvector<edge_t> tmp_in_degrees(vertex_ids.size(),
                                                                   handle_.get_stream());
                        thrust::gather(handle_.get_thrust_policy(),
                                       vertices_iter,
                                       vertices_iter + vertex_ids.size(),
                                       in_degrees->begin(),
                                       tmp_in_degrees.begin());
                        in_degrees = std::move(tmp_in_degrees);
                    }
                    else
                    {
                        rmm::device_uvector<edge_t> tmp_out_degrees(vertex_ids.size(),
                                                                    handle_.get_stream());
                        thrust::gather(handle_.get_thrust_policy(),
                                       vertices_iter,
                                       vertices_iter + vertex_ids.size(),
                                       out_degrees->begin(),
                                       tmp_out_degrees.begin());
                        out_degrees = std::move(tmp_out_degrees);
                    }

                    rocgraph::unrenumber_local_int_vertices<vertex_t>(
                        handle_,
                        vertex_ids.data(),
                        vertex_ids.size(),
                        number_map->data(),
                        graph_view.local_vertex_partition_range_first(),
                        graph_view.local_vertex_partition_range_last(),
                        do_expensive_check_);
                }
                else
                {
                    vertex_ids.resize(graph_view.local_vertex_partition_range_size(),
                                      handle_.get_stream());
                    raft::copy(vertex_ids.data(),
                               number_map->data(),
                               vertex_ids.size(),
                               handle_.get_stream());
                }

                result_ = new rocgraph::c_api::rocgraph_degrees_result_t{
                    graph_view.is_symmetric(),
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(vertex_ids,
                                                                             graph_->vertex_type_),
                    in_degrees ? new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                                     *in_degrees, graph_->edge_type_)
                               : nullptr,
                    out_degrees ? new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                                      *out_degrees, graph_->edge_type_)
                                : nullptr};
            }
        }
    };

} // namespace

extern "C" rocgraph_status
    rocgraph_in_degrees(const rocgraph_handle_t*                        handle,
                        rocgraph_graph_t*                               graph,
                        const rocgraph_type_erased_device_array_view_t* source_vertices,
                        rocgraph_bool                                   do_expensive_check,
                        rocgraph_degrees_result_t**                     result,
                        rocgraph_error_t**                              error)
{
    degrees_functor functor(handle, graph, source_vertices, true, false, do_expensive_check);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}

extern "C" rocgraph_status
    rocgraph_out_degrees(const rocgraph_handle_t*                        handle,
                         rocgraph_graph_t*                               graph,
                         const rocgraph_type_erased_device_array_view_t* source_vertices,
                         rocgraph_bool                                   do_expensive_check,
                         rocgraph_degrees_result_t**                     result,
                         rocgraph_error_t**                              error)
{
    degrees_functor functor(handle, graph, source_vertices, false, true, do_expensive_check);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}

extern "C" rocgraph_status
    rocgraph_degrees(const rocgraph_handle_t*                        handle,
                     rocgraph_graph_t*                               graph,
                     const rocgraph_type_erased_device_array_view_t* source_vertices,
                     rocgraph_bool                                   do_expensive_check,
                     rocgraph_degrees_result_t**                     result,
                     rocgraph_error_t**                              error)
{
    degrees_functor functor(handle, graph, source_vertices, true, true, do_expensive_check);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}
