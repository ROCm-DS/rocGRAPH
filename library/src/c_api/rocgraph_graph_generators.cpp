// Copyright (C) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_array.hpp"
#include "c_api/rocgraph_error.hpp"
#include "c_api/rocgraph_handle.hpp"
#include "c_api/rocgraph_random.hpp"

#include "internal/aux/rocgraph_coo_aux.h"
#include "internal/aux/rocgraph_coo_list_aux.h"
#include "internal/rocgraph_graph_generators.h"

#include "detail/utility_wrappers.hpp"
#include "graph_generators.hpp"
#include "utilities/host_scalar_comm.hpp"

#include <raft/core/handle.hpp>

namespace rocgraph
{
    namespace c_api
    {

        struct rocgraph_coo_t
        {
            std::unique_ptr<rocgraph_type_erased_device_array_t> src_{};
            std::unique_ptr<rocgraph_type_erased_device_array_t> dst_{};
            std::unique_ptr<rocgraph_type_erased_device_array_t> wgt_{};
            std::unique_ptr<rocgraph_type_erased_device_array_t> id_{};
            std::unique_ptr<rocgraph_type_erased_device_array_t> type_{};
        };

        struct rocgraph_coo_list_t
        {
            std::vector<std::unique_ptr<rocgraph_coo_t>> list_;
        };

    } // namespace c_api
} // namespace rocgraph

namespace
{

    template <typename vertex_t>
    rocgraph_status rocgraph_generate_rmat_edgelist(raft::handle_t const&   handle,
                                                    raft::random::RngState& rng_state,
                                                    rocgraph_data_type_id   vertex_dtype,
                                                    size_t                  scale,
                                                    size_t                  num_edges,
                                                    double                  a,
                                                    double                  b,
                                                    double                  c,
                                                    rocgraph_bool           clip_and_flip,
                                                    rocgraph_bool           scramble_vertex_ids,
                                                    rocgraph::c_api::rocgraph_coo_t**   result,
                                                    rocgraph::c_api::rocgraph_error_t** error)
    {
        try
        {
            auto [src, dst] = rocgraph::generate_rmat_edgelist<vertex_t>(
                handle, rng_state, scale, num_edges, a, b, c, clip_and_flip, scramble_vertex_ids);

            *result = new rocgraph::c_api::rocgraph_coo_t{
                std::make_unique<rocgraph::c_api::rocgraph_type_erased_device_array_t>(
                    src, vertex_dtype),
                std::make_unique<rocgraph::c_api::rocgraph_type_erased_device_array_t>(
                    dst, vertex_dtype),
                nullptr,
                nullptr,
                nullptr};
        }
        catch(std::exception const& ex)
        {
            *error = new rocgraph::c_api::rocgraph_error_t{ex.what()};
            return rocgraph_status_unknown_error;
        }

        return rocgraph_status_success;
    }

    template <typename vertex_t>
    rocgraph_status
        rocgraph_generate_rmat_edgelists(raft::handle_t const&                  handle,
                                         raft::random::RngState&                rng_state,
                                         rocgraph_data_type_id                  vertex_dtype,
                                         size_t                                 n_edgelists,
                                         size_t                                 min_scale,
                                         size_t                                 max_scale,
                                         size_t                                 edge_factor,
                                         rocgraph_generator_distribution        size_distribution,
                                         rocgraph_generator_distribution        edge_distribution,
                                         rocgraph_bool                          clip_and_flip,
                                         rocgraph_bool                          scramble_vertex_ids,
                                         rocgraph::c_api::rocgraph_coo_list_t** result,
                                         rocgraph::c_api::rocgraph_error_t**    error)
    {
        try
        {
            auto tuple_vector = rocgraph::generate_rmat_edgelists<vertex_t>(
                handle,
                rng_state,
                n_edgelists,
                min_scale,
                max_scale,
                edge_factor,
                static_cast<rocgraph::generator_distribution_t>(size_distribution),
                static_cast<rocgraph::generator_distribution_t>(edge_distribution),
                clip_and_flip,
                scramble_vertex_ids);

            *result = new rocgraph::c_api::rocgraph_coo_list_t;
            (*result)->list_.resize(tuple_vector.size());

            std::transform(
                tuple_vector.begin(),
                tuple_vector.end(),
                (*result)->list_.begin(),
                [vertex_dtype](auto& tuple) {
                    auto result = std::make_unique<rocgraph::c_api::rocgraph_coo_t>();

                    auto& src = std::get<0>(tuple);
                    auto& dst = std::get<1>(tuple);

                    result->src_
                        = std::make_unique<rocgraph::c_api::rocgraph_type_erased_device_array_t>(
                            src, vertex_dtype);
                    result->dst_
                        = std::make_unique<rocgraph::c_api::rocgraph_type_erased_device_array_t>(
                            dst, vertex_dtype);

                    return result;
                });

            return rocgraph_status_success;
        }
        catch(std::exception const& ex)
        {
            *error = new rocgraph::c_api::rocgraph_error_t{ex.what()};
            return rocgraph_status_unknown_error;
        }
    }

} // namespace

extern "C" rocgraph_type_erased_device_array_view_t* rocgraph_coo_get_sources(rocgraph_coo_t* coo)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_coo_t*>(coo);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->src_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_coo_get_destinations(rocgraph_coo_t* coo)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_coo_t*>(coo);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->dst_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_coo_get_edge_weights(rocgraph_coo_t* coo)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_coo_t*>(coo);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->wgt_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t* rocgraph_coo_get_edge_id(rocgraph_coo_t* coo)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_coo_t*>(coo);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->id_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t* rocgraph_coo_get_edge_type(rocgraph_coo_t* coo)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_coo_t*>(coo);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->type_->view());
}

extern "C" size_t rocgraph_coo_list_size(const rocgraph_coo_list_t* coo_list)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_coo_list_t const*>(coo_list);
    return internal_pointer->list_.size();
}

extern "C" rocgraph_coo_t* rocgraph_coo_list_element(rocgraph_coo_list_t* coo_list, size_t index)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_coo_list_t*>(coo_list);
    return reinterpret_cast<::rocgraph_coo_t*>(
        (index < internal_pointer->list_.size()) ? internal_pointer->list_[index].get() : nullptr);
}

extern "C" void rocgraph_coo_free(rocgraph_coo_t* coo)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_coo_t*>(coo);
    delete internal_pointer;
}

extern "C" void rocgraph_coo_list_free(rocgraph_coo_list_t* coo_list)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_coo_list_t*>(coo_list);
    delete internal_pointer;
}

extern "C" rocgraph_status rocgraph_generate_rmat_edgelist(const rocgraph_handle_t* handle,
                                                           rocgraph_rng_state_t*    rng_state,
                                                           size_t                   scale,
                                                           size_t                   num_edges,
                                                           double                   a,
                                                           double                   b,
                                                           double                   c,
                                                           rocgraph_bool            clip_and_flip,
                                                           rocgraph_bool      scramble_vertex_ids,
                                                           rocgraph_coo_t**   result,
                                                           rocgraph_error_t** error)
{
    auto& local_handle{*handle->get_raft_handle()};
    auto& local_rng_state{
        reinterpret_cast<rocgraph::c_api::rocgraph_rng_state_t*>(rng_state)->rng_state_};

    if(scale < 32)
    {
        return rocgraph_generate_rmat_edgelist<int32_t>(
            local_handle,
            local_rng_state,
            rocgraph_data_type_id_int32,
            scale,
            num_edges,
            a,
            b,
            c,
            clip_and_flip,
            scramble_vertex_ids,
            reinterpret_cast<rocgraph::c_api::rocgraph_coo_t**>(result),
            reinterpret_cast<rocgraph::c_api::rocgraph_error_t**>(error));
    }
    else
    {
        return rocgraph_generate_rmat_edgelist<int64_t>(
            local_handle,
            local_rng_state,
            rocgraph_data_type_id_int64,
            scale,
            num_edges,
            a,
            b,
            c,
            clip_and_flip,
            scramble_vertex_ids,
            reinterpret_cast<rocgraph::c_api::rocgraph_coo_t**>(result),
            reinterpret_cast<rocgraph::c_api::rocgraph_error_t**>(error));
    }
}

extern "C" rocgraph_status
    rocgraph_generate_rmat_edgelists(const rocgraph_handle_t*        handle,
                                     rocgraph_rng_state_t*           rng_state,
                                     size_t                          n_edgelists,
                                     size_t                          min_scale,
                                     size_t                          max_scale,
                                     size_t                          edge_factor,
                                     rocgraph_generator_distribution size_distribution,
                                     rocgraph_generator_distribution edge_distribution,
                                     rocgraph_bool                   clip_and_flip,
                                     rocgraph_bool                   scramble_vertex_ids,
                                     rocgraph_coo_list_t**           result,
                                     rocgraph_error_t**              error)
{
    auto& local_handle{*handle->get_raft_handle()};
    auto& local_rng_state{
        reinterpret_cast<rocgraph::c_api::rocgraph_rng_state_t*>(rng_state)->rng_state_};

    if(max_scale < 32)
    {
        return rocgraph_generate_rmat_edgelists<int32_t>(
            local_handle,
            local_rng_state,
            rocgraph_data_type_id_int32,
            n_edgelists,
            min_scale,
            max_scale,
            edge_factor,
            size_distribution,
            edge_distribution,
            clip_and_flip,
            scramble_vertex_ids,
            reinterpret_cast<rocgraph::c_api::rocgraph_coo_list_t**>(result),
            reinterpret_cast<rocgraph::c_api::rocgraph_error_t**>(error));
    }
    else
    {
        return rocgraph_generate_rmat_edgelists<int64_t>(
            local_handle,
            local_rng_state,
            rocgraph_data_type_id_int64,
            n_edgelists,
            min_scale,
            max_scale,
            edge_factor,
            size_distribution,
            edge_distribution,
            clip_and_flip,
            scramble_vertex_ids,
            reinterpret_cast<rocgraph::c_api::rocgraph_coo_list_t**>(result),
            reinterpret_cast<rocgraph::c_api::rocgraph_error_t**>(error));
    }
}

extern "C" rocgraph_status rocgraph_generate_edge_weights(const rocgraph_handle_t* handle,
                                                          rocgraph_rng_state_t*    rng_state,
                                                          rocgraph_coo_t*          coo,
                                                          rocgraph_data_type_id    dtype,
                                                          double                   minimum_weight,
                                                          double                   maximum_weight,
                                                          rocgraph_error_t**       error)
{
    auto& local_handle{*handle->get_raft_handle()};
    auto& local_rng_state{
        reinterpret_cast<rocgraph::c_api::rocgraph_rng_state_t*>(rng_state)->rng_state_};

    auto local_coo = reinterpret_cast<rocgraph::c_api::rocgraph_coo_t*>(coo);

    switch(dtype)
    {
    case rocgraph_data_type_id_float32:
    {
        rmm::device_uvector<float> tmp(local_coo->src_->size_, local_handle.get_stream());
        rocgraph::detail::uniform_random_fill(local_handle.get_stream(),
                                              tmp.data(),
                                              tmp.size(),
                                              static_cast<float>(minimum_weight),
                                              static_cast<float>(maximum_weight),
                                              local_rng_state);
        local_coo->wgt_
            = std::make_unique<rocgraph::c_api::rocgraph_type_erased_device_array_t>(tmp, dtype);
        break;
    }
    case rocgraph_data_type_id_float64:
    {
        rmm::device_uvector<double> tmp(local_coo->src_->size_, local_handle.get_stream());
        rocgraph::detail::uniform_random_fill(local_handle.get_stream(),
                                              tmp.data(),
                                              tmp.size(),
                                              minimum_weight,
                                              maximum_weight,
                                              local_rng_state);
        local_coo->wgt_
            = std::make_unique<rocgraph::c_api::rocgraph_type_erased_device_array_t>(tmp, dtype);
        break;
    }
    default:
    {
        *error = reinterpret_cast<::rocgraph_error_t*>(new rocgraph::c_api::rocgraph_error_t(
            "Only FLOAT and DOUBLE supported as generated edge weights"));
        return rocgraph_status_invalid_input;
    }
    }

    return rocgraph_status_success;
}

extern "C" rocgraph_status rocgraph_generate_edge_ids(const rocgraph_handle_t* handle,
                                                      rocgraph_coo_t*          coo,
                                                      rocgraph_bool            multi_gpu,
                                                      rocgraph_error_t**       error)
{
    auto& local_handle{*handle->get_raft_handle()};

    auto local_coo = reinterpret_cast<rocgraph::c_api::rocgraph_coo_t*>(coo);

    constexpr size_t int32_threshold{std::numeric_limits<int32_t>::max()};

    size_t num_edges{local_coo->src_->size_};
    size_t base_edge_id{0};

    if(multi_gpu)
    {
        auto edge_counts = rocgraph::host_scalar_allgather(
            local_handle.get_comms(), num_edges, local_handle.get_stream());
        std::vector<size_t> edge_starts(edge_counts.size());

        std::exclusive_scan(edge_counts.begin(), edge_counts.end(), edge_starts.begin(), size_t{0});

        num_edges    = edge_starts.back() + edge_counts.back();
        base_edge_id = edge_starts[local_handle.get_comms().get_rank()];
    }

    if(num_edges < int32_threshold)
    {
        rmm::device_uvector<int32_t> tmp(local_coo->src_->size_, local_handle.get_stream());

        rocgraph::detail::sequence_fill(
            local_handle.get_stream(), tmp.data(), tmp.size(), static_cast<int32_t>(base_edge_id));

        local_coo->id_ = std::make_unique<rocgraph::c_api::rocgraph_type_erased_device_array_t>(
            tmp, rocgraph_data_type_id_int32);
    }
    else
    {
        rmm::device_uvector<int64_t> tmp(local_coo->src_->size_, local_handle.get_stream());

        rocgraph::detail::sequence_fill(
            local_handle.get_stream(), tmp.data(), tmp.size(), static_cast<int64_t>(base_edge_id));

        local_coo->id_ = std::make_unique<rocgraph::c_api::rocgraph_type_erased_device_array_t>(
            tmp, rocgraph_data_type_id_int64);
    }

    return rocgraph_status_success;
}

extern "C" rocgraph_status rocgraph_generate_edge_types(const rocgraph_handle_t* handle,
                                                        rocgraph_rng_state_t*    rng_state,
                                                        rocgraph_coo_t*          coo,
                                                        int32_t                  min_edge_type,
                                                        int32_t                  max_edge_type,
                                                        rocgraph_error_t**       error)
{
    auto& local_handle{*handle->get_raft_handle()};
    auto& local_rng_state{
        reinterpret_cast<rocgraph::c_api::rocgraph_rng_state_t*>(rng_state)->rng_state_};

    auto local_coo = reinterpret_cast<rocgraph::c_api::rocgraph_coo_t*>(coo);

    rmm::device_uvector<int32_t> tmp(local_coo->src_->size_, local_handle.get_stream());
    rocgraph::detail::uniform_random_fill(local_handle.get_stream(),
                                          tmp.data(),
                                          tmp.size(),
                                          min_edge_type,
                                          max_edge_type,
                                          local_rng_state);
    local_coo->type_ = std::make_unique<rocgraph::c_api::rocgraph_type_erased_device_array_t>(
        tmp, rocgraph_data_type_id_int32);

    return rocgraph_status_success;
}
