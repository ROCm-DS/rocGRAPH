// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_abstract_functor.hpp"
#include "c_api/rocgraph_graph.hpp"
#include "c_api/rocgraph_handle.hpp"
#include "c_api/rocgraph_utils.hpp"

#include "internal/aux/rocgraph_hits_result_aux.h"
#include "internal/rocgraph_algorithms.h"

#include "algorithms.hpp"
#include "detail/shuffle_wrappers.hpp"
#include "detail/utility_wrappers.hpp"
#include "graph_functions.hpp"

#include <optional>

namespace rocgraph
{
    namespace c_api
    {

        struct rocgraph_hits_result_t
        {
            rocgraph_type_erased_device_array_t* vertex_ids_;
            rocgraph_type_erased_device_array_t* hubs_;
            rocgraph_type_erased_device_array_t* authorities_;
            double                               hub_score_differences_;
            size_t                               number_of_iterations_;
        };

    } // namespace c_api
} // namespace rocgraph

namespace
{

    struct hits_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&              handle_;
        rocgraph::c_api::rocgraph_graph_t* graph_;
        double                             epsilon_;
        size_t                             max_iterations_;
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*
            initial_hubs_guess_vertices_;
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* initial_hubs_guess_values_;
        bool                                                             normalize_;
        bool                                                             do_expensive_check_;
        rocgraph::c_api::rocgraph_hits_result_t*                         result_{};

        hits_functor(::rocgraph_handle_t const*                        handle,
                     ::rocgraph_graph_t*                               graph,
                     double                                            epsilon,
                     size_t                                            max_iterations,
                     ::rocgraph_type_erased_device_array_view_t const* initial_hubs_guess_vertices,
                     ::rocgraph_type_erased_device_array_view_t const* initial_hubs_guess_values,
                     bool                                              normalize,
                     bool                                              do_expensive_check)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , epsilon_(epsilon)
            , max_iterations_(max_iterations)
            , initial_hubs_guess_vertices_(
                  reinterpret_cast<
                      rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                      initial_hubs_guess_vertices))
            , initial_hubs_guess_values_(
                  reinterpret_cast<
                      rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                      initial_hubs_guess_values))
            , normalize_(normalize)
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
                // HITS expects store_transposed == true
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

                auto number_map
                    = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

                rmm::device_uvector<weight_t> hubs(graph_view.local_vertex_partition_range_size(),
                                                   handle_.get_stream());
                rmm::device_uvector<weight_t> authorities(
                    graph_view.local_vertex_partition_range_size(), handle_.get_stream());
                weight_t hub_score_differences{0};
                size_t   number_of_iterations{0};

                if(initial_hubs_guess_vertices_ != nullptr)
                {
                    rmm::device_uvector<vertex_t> guess_vertices(
                        initial_hubs_guess_vertices_->size_, handle_.get_stream());
                    rmm::device_uvector<weight_t> guess_values(initial_hubs_guess_values_->size_,
                                                               handle_.get_stream());

                    raft::copy(guess_vertices.data(),
                               initial_hubs_guess_vertices_->as_type<vertex_t>(),
                               guess_vertices.size(),
                               handle_.get_stream());
                    raft::copy(guess_values.data(),
                               initial_hubs_guess_values_->as_type<weight_t>(),
                               guess_values.size(),
                               handle_.get_stream());

                    hubs
                        = rocgraph::detail::collect_local_vertex_values_from_ext_vertex_value_pairs<
                            vertex_t,
                            weight_t,
                            multi_gpu>(handle_,
                                       std::move(guess_vertices),
                                       std::move(guess_values),
                                       *number_map,
                                       graph_view.local_vertex_partition_range_first(),
                                       graph_view.local_vertex_partition_range_last(),
                                       weight_t{0},
                                       do_expensive_check_);
                }

                std::tie(hub_score_differences, number_of_iterations)
                    = rocgraph::hits<vertex_t, edge_t, weight_t, multi_gpu>(
                        handle_,
                        graph_view,
                        hubs.data(),
                        authorities.data(),
                        epsilon_,
                        max_iterations_,
                        (initial_hubs_guess_vertices_ != nullptr),
                        normalize_,
                        do_expensive_check_);

                rmm::device_uvector<vertex_t> vertex_ids(
                    graph_view.local_vertex_partition_range_size(), handle_.get_stream());
                raft::copy(
                    vertex_ids.data(), number_map->data(), vertex_ids.size(), handle_.get_stream());

                result_ = new rocgraph::c_api::rocgraph_hits_result_t{
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(vertex_ids,
                                                                             graph_->vertex_type_),
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(hubs,
                                                                             graph_->weight_type_),
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(authorities,
                                                                             graph_->weight_type_),
                    hub_score_differences,
                    number_of_iterations};
            }
        }
    };

} // namespace

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_hits_result_get_vertices(rocgraph_hits_result_t* result)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_hits_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->vertex_ids_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_hits_result_get_hubs(rocgraph_hits_result_t* result)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_hits_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->hubs_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_hits_result_get_authorities(rocgraph_hits_result_t* result)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_hits_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->authorities_->view());
}

extern "C" double rocgraph_hits_result_get_hub_score_differences(rocgraph_hits_result_t* result)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_hits_result_t*>(result);
    return internal_pointer->hub_score_differences_;
}

extern "C" size_t rocgraph_hits_result_get_number_of_iterations(rocgraph_hits_result_t* result)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_hits_result_t*>(result);
    return internal_pointer->number_of_iterations_;
}

extern "C" void rocgraph_hits_result_free(rocgraph_hits_result_t* result)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_hits_result_t*>(result);
    delete internal_pointer->vertex_ids_;
    delete internal_pointer->hubs_;
    delete internal_pointer->authorities_;
    delete internal_pointer;
}

extern "C" rocgraph_status
    rocgraph_hits(const rocgraph_handle_t*                        handle,
                  rocgraph_graph_t*                               graph,
                  double                                          epsilon,
                  size_t                                          max_iterations,
                  const rocgraph_type_erased_device_array_view_t* initial_hubs_guess_vertices,
                  const rocgraph_type_erased_device_array_view_t* initial_hubs_guess_values,
                  rocgraph_bool                                   normalize,
                  rocgraph_bool                                   do_expensive_check,
                  rocgraph_hits_result_t**                        result,
                  rocgraph_error_t**                              error)
{
    hits_functor functor(handle,
                         graph,
                         epsilon,
                         max_iterations,
                         initial_hubs_guess_vertices,
                         initial_hubs_guess_values,
                         normalize,
                         do_expensive_check);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}
