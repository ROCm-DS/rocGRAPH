// Copyright (C) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_random.hpp"

#include "c_api/rocgraph_abstract_functor.hpp"
#include "c_api/rocgraph_error.hpp"
#include "c_api/rocgraph_graph.hpp"
#include "c_api/rocgraph_handle.hpp"
#include "c_api/rocgraph_utils.hpp"

#include "detail/shuffle_wrappers.hpp"
#include "internal/aux/rocgraph_rng_state_aux.h"
#include "internal/rocgraph_algorithms.h"
#include "utilities/host_scalar_comm.hpp"

namespace
{

    struct select_random_vertices_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                                 handle_;
        rocgraph::c_api::rocgraph_graph_t const*              graph_{};
        rocgraph::c_api::rocgraph_rng_state_t*                rng_state_{nullptr};
        size_t                                                num_vertices_{};
        rocgraph::c_api::rocgraph_type_erased_device_array_t* result_{};

        select_random_vertices_functor(rocgraph_handle_t const* handle,
                                       rocgraph_graph_t const*  graph,
                                       rocgraph_rng_state_t*    rng_state,
                                       size_t                   num_vertices)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t const*>(graph))
            , rng_state_(reinterpret_cast<rocgraph::c_api::rocgraph_rng_state_t*>(rng_state))
            , num_vertices_(num_vertices)
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
                auto graph = reinterpret_cast<
                    rocgraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>*>(
                    graph_->graph_);

                auto graph_view = graph->view();
                auto number_map
                    = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

                rmm::device_uvector<vertex_t> local_vertices(0, handle_.get_stream());

                local_vertices = rocgraph::select_random_vertices(
                    handle_,
                    graph_view,
                    std::optional<raft::device_span<vertex_t const>>{std::nullopt},
                    rng_state_->rng_state_,
                    num_vertices_,
                    false,
                    false);

                rocgraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
                    handle_,
                    local_vertices.data(),
                    local_vertices.size(),
                    number_map->data(),
                    graph_view.vertex_partition_range_lasts(),
                    false);

                result_ = new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                    local_vertices, graph_->vertex_type_);
            }
        }
    };

} // namespace

extern "C" rocgraph_status rocgraph_rng_state_create(const rocgraph_handle_t* handle,
                                                     uint64_t                 seed,
                                                     rocgraph_rng_state_t**   state,
                                                     rocgraph_error_t**       error)
{
    *state = nullptr;
    *error = nullptr;

    try
    {
        if(!handle)
        {
            *error = reinterpret_cast<rocgraph_error_t*>(
                new rocgraph::c_api::rocgraph_error_t{"invalid resource handle"});
            return rocgraph_status_invalid_handle;
        }

        auto p_handle = handle;

        if(p_handle->get_raft_handle()->comms_initialized())
        {
            // need to verify that every seed is different
            auto seed_v
                = rocgraph::host_scalar_allgather(p_handle->get_raft_handle()->get_comms(),
                                                  seed,
                                                  p_handle->get_raft_handle()->get_stream());
            std::sort(seed_v.begin(), seed_v.end());
            if(std::unique(seed_v.begin(), seed_v.end()) != seed_v.end())
            {
                *error = reinterpret_cast<rocgraph_error_t*>(
                    new rocgraph::c_api::rocgraph_error_t{"seed must be different on each GPU"});
                return rocgraph_status_invalid_input;
            }
        }

        *state = reinterpret_cast<rocgraph_rng_state_t*>(
            new rocgraph::c_api::rocgraph_rng_state_t{raft::random::RngState{seed}});
        return rocgraph_status_success;
    }
    catch(std::exception const& ex)
    {
        *error
            = reinterpret_cast<rocgraph_error_t*>(new rocgraph::c_api::rocgraph_error_t{ex.what()});
        return rocgraph_status_unknown_error;
    }
}

extern "C" void rocgraph_rng_state_free(rocgraph_rng_state_t* p)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_rng_state_t*>(p);
    delete internal_pointer;
}

extern "C" rocgraph_status
    rocgraph_select_random_vertices(const rocgraph_handle_t*              handle,
                                    const rocgraph_graph_t*               graph,
                                    rocgraph_rng_state_t*                 rng_state,
                                    size_t                                num_vertices,
                                    rocgraph_type_erased_device_array_t** vertices,
                                    rocgraph_error_t**                    error)
{
    select_random_vertices_functor functor(handle, graph, rng_state, num_vertices);

    return rocgraph::c_api::run_algorithm(graph, functor, vertices, error);
}
