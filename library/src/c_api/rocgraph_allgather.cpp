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
#include "internal/rocgraph_graph_functions.h"
#include "internal/rocgraph_graph_generators.h"

#include "algorithms.hpp"
#include "detail/collect_comm_wrapper.hpp"
#include "detail/shuffle_wrappers.hpp"
#include "detail/utility_wrappers.hpp"
#include "graph_functions.hpp"

namespace
{

    struct create_allgather_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                                            handle_;
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* src_;
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* dst_;
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* weights_;
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* edge_ids_;
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* edge_type_ids_;
        rocgraph::c_api::rocgraph_induced_subgraph_result_t*             result_{};

        create_allgather_functor(
            raft::handle_t const&                                            handle,
            rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* src,
            rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* dst,
            rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* weights,
            rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* edge_ids,
            rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* edge_type_ids)
            : abstract_functor()
            , handle_(handle)
            , src_(src)
            , dst_(dst)
            , weights_(weights)
            , edge_ids_(edge_ids)
            , edge_type_ids_(edge_type_ids)
        {
        }

        template <typename vertex_t,
                  typename edge_t,
                  typename weight_t,
                  typename edge_type_id_t,
                  bool store_transposed,
                  bool multi_gpu>
        void operator()()
        {
            std::optional<rmm::device_uvector<vertex_t>> edgelist_srcs{std::nullopt};
            if(src_)
            {
                edgelist_srcs = rmm::device_uvector<vertex_t>(src_->size_, handle_.get_stream());
                raft::copy(edgelist_srcs->data(),
                           src_->as_type<vertex_t>(),
                           src_->size_,
                           handle_.get_stream());
            }

            std::optional<rmm::device_uvector<vertex_t>> edgelist_dsts{std::nullopt};
            if(dst_)
            {
                edgelist_dsts = rmm::device_uvector<vertex_t>(dst_->size_, handle_.get_stream());
                raft::copy(edgelist_dsts->data(),
                           dst_->as_type<vertex_t>(),
                           dst_->size_,
                           handle_.get_stream());
            }

            std::optional<rmm::device_uvector<weight_t>> edgelist_weights{std::nullopt};
            if(weights_)
            {
                edgelist_weights
                    = rmm::device_uvector<weight_t>(weights_->size_, handle_.get_stream());
                raft::copy(edgelist_weights->data(),
                           weights_->as_type<weight_t>(),
                           weights_->size_,
                           handle_.get_stream());
            }

            std::optional<rmm::device_uvector<edge_t>> edgelist_ids{std::nullopt};
            if(edge_ids_)
            {
                edgelist_ids = rmm::device_uvector<edge_t>(edge_ids_->size_, handle_.get_stream());
                raft::copy(edgelist_ids->data(),
                           edge_ids_->as_type<edge_t>(),
                           edge_ids_->size_,
                           handle_.get_stream());
            }

            std::optional<rmm::device_uvector<edge_type_id_t>> edgelist_type_ids{std::nullopt};
            if(edge_type_ids_)
            {
                edgelist_type_ids = rmm::device_uvector<edge_type_id_t>(edge_type_ids_->size_,
                                                                        handle_.get_stream());
                raft::copy(edgelist_type_ids->data(),
                           edge_type_ids_->as_type<edge_type_id_t>(),
                           edge_type_ids_->size_,
                           handle_.get_stream());
            }

            auto& comm = handle_.get_comms();

            if(edgelist_srcs)
            {
                edgelist_srcs = rocgraph::detail::device_allgatherv(
                    handle_,
                    comm,
                    raft::device_span<vertex_t const>(edgelist_srcs->data(),
                                                      edgelist_srcs->size()));
            }

            if(edgelist_dsts)
            {
                edgelist_dsts = rocgraph::detail::device_allgatherv(
                    handle_,
                    comm,
                    raft::device_span<vertex_t const>(edgelist_dsts->data(),
                                                      edgelist_dsts->size()));
            }

            rmm::device_uvector<size_t> edge_offsets(2, handle_.get_stream());

            std::vector<size_t> h_edge_offsets{
                {0, edgelist_srcs ? edgelist_srcs->size() : edgelist_weights->size()}};
            raft::update_device(edge_offsets.data(),
                                h_edge_offsets.data(),
                                h_edge_offsets.size(),
                                handle_.get_stream());

            rocgraph::c_api::rocgraph_induced_subgraph_result_t* result = NULL;

            if(edgelist_weights)
            {
                edgelist_weights = rocgraph::detail::device_allgatherv(
                    handle_,
                    comm,
                    raft::device_span<weight_t const>(edgelist_weights->data(),
                                                      edgelist_weights->size()));
            }

            if(edgelist_ids)
            {
                edgelist_ids = rocgraph::detail::device_allgatherv(
                    handle_,
                    comm,
                    raft::device_span<edge_t const>(edgelist_ids->data(), edgelist_ids->size()));
            }

            if(edgelist_type_ids)
            {
                edgelist_type_ids = rocgraph::detail::device_allgatherv(
                    handle_,
                    comm,
                    raft::device_span<edge_type_id_t const>(edgelist_type_ids->data(),
                                                            edgelist_type_ids->size()));
            }

            result = new rocgraph::c_api::rocgraph_induced_subgraph_result_t{
                edgelist_srcs ? new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                                    *edgelist_srcs, src_->type_)
                              : NULL,
                edgelist_dsts ? new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                                    *edgelist_dsts, dst_->type_)
                              : NULL,
                edgelist_weights ? new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                                       *edgelist_weights, weights_->type_)
                                 : NULL,
                edgelist_ids ? new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                                   *edgelist_ids, edge_ids_->type_)
                             : NULL,
                edgelist_type_ids ? new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                                        *edgelist_type_ids, edge_type_ids_->type_)
                                  : NULL,
                new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                    edge_offsets, rocgraph_data_type_id_size_t)};

            result_
                = reinterpret_cast<rocgraph::c_api::rocgraph_induced_subgraph_result_t*>(result);
        }
    };

} // namespace

extern "C" rocgraph_status
    rocgraph_allgather(const rocgraph_handle_t*                        handle,
                       const rocgraph_type_erased_device_array_view_t* src,
                       const rocgraph_type_erased_device_array_view_t* dst,
                       const rocgraph_type_erased_device_array_view_t* weights,
                       const rocgraph_type_erased_device_array_view_t* edge_ids,
                       const rocgraph_type_erased_device_array_view_t* edge_type_ids,
                       rocgraph_induced_subgraph_result_t**            edgelist,
                       rocgraph_error_t**                              error)
{
    *edgelist = nullptr;
    *error    = nullptr;

    auto p_handle = handle;
    auto p_src
        = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(src);
    auto p_dst
        = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(dst);
    auto p_weights
        = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
            weights);

    auto p_edge_ids
        = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
            edge_ids);

    auto p_edge_type_ids
        = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
            edge_type_ids);

    CAPI_EXPECTS((dst == nullptr) || (src == nullptr) || p_src->size_ == p_dst->size_,
                 rocgraph_status_invalid_input,
                 "Invalid input arguments: src size != dst size.",
                 *error);
    CAPI_EXPECTS((dst == nullptr) || (src == nullptr) || p_src->type_ == p_dst->type_,
                 rocgraph_status_invalid_input,
                 "Invalid input arguments: src type != dst type.",
                 *error);

    CAPI_EXPECTS((weights == nullptr) || (src == nullptr) || (p_weights->size_ == p_src->size_),
                 rocgraph_status_invalid_input,
                 "Invalid input arguments: src size != weights size.",
                 *error);

    rocgraph_data_type_id vertex_type;
    rocgraph_data_type_id edge_type;
    rocgraph_data_type_id weight_type;
    rocgraph_data_type_id edge_type_id_type;

    if(src != nullptr)
    {
        vertex_type = p_src->type_;
    }
    else
    {
        vertex_type = rocgraph_data_type_id_int32;
    }

    if(weights != nullptr)
    {
        weight_type = p_weights->type_;
    }
    else
    {
        weight_type = rocgraph_data_type_id_float32;
    }

    if(edge_ids != nullptr)
    {
        edge_type = p_edge_ids->type_;
    }
    else
    {
        edge_type = rocgraph_data_type_id_int32;
    }

    if(edge_type_ids != nullptr)
    {
        edge_type_id_type = p_edge_type_ids->type_;
    }
    else
    {
        edge_type_id_type = rocgraph_data_type_id_int32;
    }

    if(src != nullptr)
    {
        CAPI_EXPECTS((edge_ids == nullptr) || (p_edge_ids->size_ == p_src->size_),
                     rocgraph_status_invalid_input,
                     "Invalid input arguments: src size != edge id prop size",
                     *error);

        CAPI_EXPECTS((edge_type_ids == nullptr) || (p_edge_type_ids->size_ == p_src->size_),
                     rocgraph_status_invalid_input,
                     "Invalid input arguments: src size != edge type prop size",
                     *error);
    }

    constexpr bool multi_gpu        = false;
    constexpr bool store_transposed = false;

    ::create_allgather_functor functor(
        *p_handle->get_raft_handle(), p_src, p_dst, p_weights, p_edge_ids, p_edge_type_ids);

    try
    {
        rocgraph::c_api::vertex_dispatcher(vertex_type,
                                           edge_type,
                                           weight_type,
                                           edge_type_id_type,
                                           store_transposed,
                                           multi_gpu,
                                           functor);

        if(functor.status_ != rocgraph_status_success)
        {
            *error = reinterpret_cast<rocgraph_error_t*>(functor.error_.release());
            return functor.status_;
        }

        *edgelist = reinterpret_cast<rocgraph_induced_subgraph_result_t*>(functor.result_);
    }
    catch(std::exception const& ex)
    {
        *error
            = reinterpret_cast<rocgraph_error_t*>(new rocgraph::c_api::rocgraph_error_t{ex.what()});
        return rocgraph_status_unknown_error;
    }

    return rocgraph_status_success;
}
