// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_abstract_functor.hpp"
#include "c_api/rocgraph_array.hpp"
#include "c_api/rocgraph_error.hpp"
#include "c_api/rocgraph_generic_cascaded_dispatch.hpp"
#include "c_api/rocgraph_graph.hpp"
#include "c_api/rocgraph_graph_helper.hpp"
#include "c_api/rocgraph_handle.hpp"

#include "internal/aux/rocgraph_graph_aux.h"

#include "detail/utility_wrappers.hpp"
#include "graph_functions.hpp"

#include <limits>

namespace
{

    struct create_graph_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                                            handle_;
        rocgraph_graph_properties_t const*                               properties_;
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* vertices_;
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* src_;
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* dst_;
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* weights_;
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* edge_ids_;
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* edge_type_ids_;
        rocgraph_bool                                                    renumber_;
        rocgraph_bool                                                    drop_self_loops_;
        rocgraph_bool                                                    drop_multi_edges_;
        rocgraph_bool                                                    do_expensive_check_;
        rocgraph_data_type_id                                            edge_type_;
        rocgraph::c_api::rocgraph_graph_t*                               result_{};

        create_graph_functor(
            raft::handle_t const&                                            handle,
            rocgraph_graph_properties_t const*                               properties,
            rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* vertices,
            rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* src,
            rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* dst,
            rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* weights,
            rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* edge_ids,
            rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* edge_type_ids,
            rocgraph_bool                                                    renumber,
            rocgraph_bool                                                    drop_self_loops,
            rocgraph_bool                                                    drop_multi_edges,
            rocgraph_bool                                                    do_expensive_check,
            rocgraph_data_type_id                                            edge_type)
            : abstract_functor()
            , properties_(properties)
            , handle_(handle)
            , vertices_(vertices)
            , src_(src)
            , dst_(dst)
            , weights_(weights)
            , edge_ids_(edge_ids)
            , edge_type_ids_(edge_type_ids)
            , renumber_(renumber)
            , drop_self_loops_(drop_self_loops)
            , drop_multi_edges_(drop_multi_edges)
            , do_expensive_check_(do_expensive_check)
            , edge_type_(edge_type)
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
            if constexpr(multi_gpu || !rocgraph::is_candidate<vertex_t, edge_t, weight_t>::value)
            {
                unsupported();
            }
            else
            {
                if(do_expensive_check_)
                {
                    // FIXME:  Need an implementation here.
                }

                std::optional<rmm::device_uvector<vertex_t>> new_number_map;

                std::optional<rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                    weight_t>>
                    new_edge_weights{std::nullopt};

                std::optional<rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                    edge_t>>
                    new_edge_ids{std::nullopt};

                std::optional<rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                    edge_type_id_t>>
                    new_edge_types{std::nullopt};

                std::optional<rmm::device_uvector<vertex_t>> vertex_list
                    = vertices_ ? std::make_optional(rmm::device_uvector<vertex_t>(
                                      vertices_->size_, handle_.get_stream()))
                                : std::nullopt;

                if(vertex_list)
                {
                    raft::copy<vertex_t>(vertex_list->data(),
                                         vertices_->as_type<vertex_t>(),
                                         vertices_->size_,
                                         handle_.get_stream());
                }

                rmm::device_uvector<vertex_t> edgelist_srcs(src_->size_, handle_.get_stream());
                rmm::device_uvector<vertex_t> edgelist_dsts(dst_->size_, handle_.get_stream());

                raft::copy<vertex_t>(edgelist_srcs.data(),
                                     src_->as_type<vertex_t>(),
                                     src_->size_,
                                     handle_.get_stream());
                raft::copy<vertex_t>(edgelist_dsts.data(),
                                     dst_->as_type<vertex_t>(),
                                     dst_->size_,
                                     handle_.get_stream());

                std::optional<rmm::device_uvector<weight_t>> edgelist_weights
                    = weights_ ? std::make_optional(rmm::device_uvector<weight_t>(
                                     weights_->size_, handle_.get_stream()))
                               : std::nullopt;

                if(edgelist_weights)
                {
                    raft::copy<weight_t>(edgelist_weights->data(),
                                         weights_->as_type<weight_t>(),
                                         weights_->size_,
                                         handle_.get_stream());
                }

                std::optional<rmm::device_uvector<edge_t>> edgelist_edge_ids
                    = edge_ids_ ? std::make_optional(rmm::device_uvector<edge_t>(
                                      edge_ids_->size_, handle_.get_stream()))
                                : std::nullopt;

                if(edgelist_edge_ids)
                {
                    raft::copy<edge_t>(edgelist_edge_ids->data(),
                                       edge_ids_->as_type<edge_t>(),
                                       edge_ids_->size_,
                                       handle_.get_stream());
                }

                std::optional<rmm::device_uvector<edge_type_id_t>> edgelist_edge_types
                    = edge_type_ids_ ? std::make_optional(rmm::device_uvector<edge_type_id_t>(
                                           edge_type_ids_->size_, handle_.get_stream()))
                                     : std::nullopt;

                if(edgelist_edge_types)
                {
                    raft::copy<edge_type_id_t>(edgelist_edge_types->data(),
                                               edge_type_ids_->as_type<edge_type_id_t>(),
                                               edge_type_ids_->size_,
                                               handle_.get_stream());
                }

                auto graph
                    = new rocgraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>(handle_);

                rmm::device_uvector<vertex_t>* number_map
                    = new rmm::device_uvector<vertex_t>(0, handle_.get_stream());

                auto edge_weights = new rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                    weight_t>(handle_);

                auto edge_ids = new rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                    edge_t>(handle_);

                auto edge_types = new rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                    edge_type_id_t>(handle_);

                if(drop_self_loops_)
                {
                    std::tie(edgelist_srcs,
                             edgelist_dsts,
                             edgelist_weights,
                             edgelist_edge_ids,
                             edgelist_edge_types)
                        = rocgraph::remove_self_loops(handle_,
                                                      std::move(edgelist_srcs),
                                                      std::move(edgelist_dsts),
                                                      std::move(edgelist_weights),
                                                      std::move(edgelist_edge_ids),
                                                      std::move(edgelist_edge_types));
                }

                if(drop_multi_edges_)
                {
                    std::tie(edgelist_srcs,
                             edgelist_dsts,
                             edgelist_weights,
                             edgelist_edge_ids,
                             edgelist_edge_types)
                        = rocgraph::remove_multi_edges(
                            handle_,
                            std::move(edgelist_srcs),
                            std::move(edgelist_dsts),
                            std::move(edgelist_weights),
                            std::move(edgelist_edge_ids),
                            std::move(edgelist_edge_types),
                            properties_->is_symmetric
                                ? true /* keep minimum weight edges to maintain symmetry */
                                : false);
                }

                std::tie(*graph, new_edge_weights, new_edge_ids, new_edge_types, new_number_map)
                    = rocgraph::create_graph_from_edgelist<vertex_t,
                                                           edge_t,
                                                           weight_t,
                                                           edge_t,
                                                           edge_type_id_t,
                                                           store_transposed,
                                                           multi_gpu>(
                        handle_,
                        std::move(vertex_list),
                        std::move(edgelist_srcs),
                        std::move(edgelist_dsts),
                        std::move(edgelist_weights),
                        std::move(edgelist_edge_ids),
                        std::move(edgelist_edge_types),
                        rocgraph::graph_properties_t{properties_->is_symmetric,
                                                     properties_->is_multigraph},
                        renumber_,
                        do_expensive_check_);

                if(renumber_)
                {
                    *number_map = std::move(new_number_map.value());
                }
                else
                {
                    number_map->resize(graph->number_of_vertices(), handle_.get_stream());
                    rocgraph::detail::sequence_fill(
                        handle_.get_stream(),
                        number_map->data(),
                        number_map->size(),
                        graph->view().local_vertex_partition_range_first());
                }

                if(new_edge_weights)
                {
                    *edge_weights = std::move(new_edge_weights.value());
                }
                if(new_edge_ids)
                {
                    *edge_ids = std::move(new_edge_ids.value());
                }
                if(new_edge_types)
                {
                    *edge_types = std::move(new_edge_types.value());
                }

                // Set up return
                auto result = new rocgraph::c_api::rocgraph_graph_t{
                    src_->type_,
                    edge_type_,
                    weights_ ? weights_->type_ : rocgraph_data_type_id_float32,
                    edge_type_ids_ ? edge_type_ids_->type_ : rocgraph_data_type_id_int32,
                    store_transposed,
                    multi_gpu,
                    graph,
                    number_map,
                    new_edge_weights ? edge_weights : nullptr,
                    new_edge_ids ? edge_ids : nullptr,
                    new_edge_types ? edge_types : nullptr};

                result_ = reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(result);
            }
        }
    };

    struct create_graph_csr_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                                            handle_;
        rocgraph_graph_properties_t const*                               properties_;
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* offsets_;
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* indices_;
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* weights_;
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* edge_ids_;
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* edge_type_ids_;
        rocgraph_bool                                                    renumber_;
        rocgraph_bool                                                    do_expensive_check_;
        rocgraph::c_api::rocgraph_graph_t*                               result_{};

        create_graph_csr_functor(
            raft::handle_t const&                                            handle,
            rocgraph_graph_properties_t const*                               properties,
            rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* offsets,
            rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* indices,
            rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* weights,
            rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* edge_ids,
            rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* edge_type_ids,
            rocgraph_bool                                                    renumber,
            rocgraph_bool                                                    do_expensive_check)
            : abstract_functor()
            , properties_(properties)
            , handle_(handle)
            , offsets_(offsets)
            , indices_(indices)
            , weights_(weights)
            , edge_ids_(edge_ids)
            , edge_type_ids_(edge_type_ids)
            , renumber_(renumber)
            , do_expensive_check_(do_expensive_check)
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
            if constexpr(multi_gpu || !rocgraph::is_candidate<vertex_t, edge_t, weight_t>::value)
            {
                unsupported();
            }
            else
            {
                if(do_expensive_check_)
                {
                    // FIXME:  Need an implementation here.
                }

                std::optional<rmm::device_uvector<vertex_t>> new_number_map;

                std::optional<rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                    weight_t>>
                    new_edge_weights{std::nullopt};

                std::optional<rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                    edge_t>>
                    new_edge_ids{std::nullopt};

                std::optional<rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                    edge_type_id_t>>
                    new_edge_types{std::nullopt};

                std::optional<rmm::device_uvector<vertex_t>> vertex_list = std::make_optional(
                    rmm::device_uvector<vertex_t>(offsets_->size_ - 1, handle_.get_stream()));

                rocgraph::detail::sequence_fill(
                    handle_.get_stream(), vertex_list->data(), vertex_list->size(), vertex_t{0});

                rmm::device_uvector<vertex_t> edgelist_srcs(0, handle_.get_stream());
                rmm::device_uvector<vertex_t> edgelist_dsts(indices_->size_, handle_.get_stream());

                edgelist_srcs = rocgraph::c_api::expand_sparse_offsets(
                    raft::device_span<edge_t const>{offsets_->as_type<edge_t>(), offsets_->size_},
                    vertex_t{0},
                    handle_.get_stream());
                raft::copy<vertex_t>(edgelist_dsts.data(),
                                     indices_->as_type<vertex_t>(),
                                     indices_->size_,
                                     handle_.get_stream());

                std::optional<rmm::device_uvector<weight_t>> edgelist_weights
                    = weights_ ? std::make_optional(rmm::device_uvector<weight_t>(
                                     weights_->size_, handle_.get_stream()))
                               : std::nullopt;

                if(edgelist_weights)
                {
                    raft::copy<weight_t>(edgelist_weights->data(),
                                         weights_->as_type<weight_t>(),
                                         weights_->size_,
                                         handle_.get_stream());
                }

                std::optional<
                    std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_id_t>>>
                    edgelist_edge_tuple{};

                std::optional<rmm::device_uvector<edge_t>> edgelist_edge_ids
                    = edge_ids_ ? std::make_optional(rmm::device_uvector<edge_t>(
                                      edge_ids_->size_, handle_.get_stream()))
                                : std::nullopt;

                if(edgelist_edge_ids)
                {
                    raft::copy<edge_t>(edgelist_edge_ids->data(),
                                       edge_ids_->as_type<edge_t>(),
                                       edge_ids_->size_,
                                       handle_.get_stream());
                }

                std::optional<rmm::device_uvector<edge_type_id_t>> edgelist_edge_types
                    = edge_type_ids_ ? std::make_optional(rmm::device_uvector<edge_type_id_t>(
                                           edge_type_ids_->size_, handle_.get_stream()))
                                     : std::nullopt;

                if(edgelist_edge_types)
                {
                    raft::copy<edge_type_id_t>(edgelist_edge_types->data(),
                                               edge_type_ids_->as_type<edge_type_id_t>(),
                                               edge_type_ids_->size_,
                                               handle_.get_stream());
                }

                auto graph
                    = new rocgraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>(handle_);

                rmm::device_uvector<vertex_t>* number_map
                    = new rmm::device_uvector<vertex_t>(0, handle_.get_stream());

                auto edge_weights = new rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                    weight_t>(handle_);

                auto edge_ids = new rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                    edge_t>(handle_);

                auto edge_types = new rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                    edge_type_id_t>(handle_);

                std::tie(*graph, new_edge_weights, new_edge_ids, new_edge_types, new_number_map)
                    = rocgraph::create_graph_from_edgelist<vertex_t,
                                                           edge_t,
                                                           weight_t,
                                                           edge_t,
                                                           edge_type_id_t,
                                                           store_transposed,
                                                           multi_gpu>(
                        handle_,
                        std::move(vertex_list),
                        std::move(edgelist_srcs),
                        std::move(edgelist_dsts),
                        std::move(edgelist_weights),
                        std::move(edgelist_edge_ids),
                        std::move(edgelist_edge_types),
                        rocgraph::graph_properties_t{properties_->is_symmetric,
                                                     properties_->is_multigraph},
                        renumber_,
                        do_expensive_check_);

                if(renumber_)
                {
                    *number_map = std::move(new_number_map.value());
                }
                else
                {
                    number_map->resize(graph->number_of_vertices(), handle_.get_stream());
                    rocgraph::detail::sequence_fill(
                        handle_.get_stream(),
                        number_map->data(),
                        number_map->size(),
                        graph->view().local_vertex_partition_range_first());
                }

                if(new_edge_weights)
                {
                    *edge_weights = std::move(new_edge_weights.value());
                }
                if(new_edge_ids)
                {
                    *edge_ids = std::move(new_edge_ids.value());
                }
                if(new_edge_types)
                {
                    *edge_types = std::move(new_edge_types.value());
                }

                // Set up return
                auto result = new rocgraph::c_api::rocgraph_graph_t{
                    indices_->type_,
                    offsets_->type_,
                    weights_ ? weights_->type_ : rocgraph_data_type_id_float32,
                    edge_type_ids_ ? edge_type_ids_->type_ : rocgraph_data_type_id_int32,
                    store_transposed,
                    multi_gpu,
                    graph,
                    number_map,
                    new_edge_weights ? edge_weights : nullptr,
                    new_edge_ids ? edge_ids : nullptr,
                    new_edge_types ? edge_types : nullptr};

                result_ = reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(result);
            }
        }
    };

    struct destroy_graph_functor : public rocgraph::c_api::abstract_functor
    {
        void* graph_;
        void* number_map_;
        void* edge_weights_;
        void* edge_ids_;
        void* edge_types_;

        destroy_graph_functor(
            void* graph, void* number_map, void* edge_weights, void* edge_ids, void* edge_types)
            : abstract_functor()
            , graph_(graph)
            , number_map_(number_map)
            , edge_weights_(edge_weights)
            , edge_ids_(edge_ids)
            , edge_types_(edge_types)
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
            auto internal_graph_pointer = reinterpret_cast<
                rocgraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>*>(graph_);

            delete internal_graph_pointer;

            auto internal_number_map_pointer
                = reinterpret_cast<rmm::device_uvector<vertex_t>*>(number_map_);

            delete internal_number_map_pointer;

            auto internal_edge_weight_pointer = reinterpret_cast<rocgraph::edge_property_t<
                rocgraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                weight_t>*>(edge_weights_);
            if(internal_edge_weight_pointer)
            {
                delete internal_edge_weight_pointer;
            }

            auto internal_edge_id_pointer = reinterpret_cast<rocgraph::edge_property_t<
                rocgraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                edge_t>*>(edge_ids_);
            if(internal_edge_id_pointer)
            {
                delete internal_edge_id_pointer;
            }

            auto internal_edge_type_pointer = reinterpret_cast<rocgraph::edge_property_t<
                rocgraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                edge_type_id_t>*>(edge_types_);
            if(internal_edge_type_pointer)
            {
                delete internal_edge_type_pointer;
            }
        }
    };

} // namespace

extern "C" rocgraph_status
    rocgraph_graph_create_sg(const rocgraph_handle_t*                        handle,
                             const rocgraph_graph_properties_t*              properties,
                             const rocgraph_type_erased_device_array_view_t* vertices,
                             const rocgraph_type_erased_device_array_view_t* src,
                             const rocgraph_type_erased_device_array_view_t* dst,
                             const rocgraph_type_erased_device_array_view_t* weights,
                             const rocgraph_type_erased_device_array_view_t* edge_ids,
                             const rocgraph_type_erased_device_array_view_t* edge_type_ids,
                             rocgraph_bool                                   store_transposed,
                             rocgraph_bool                                   renumber,
                             rocgraph_bool                                   drop_self_loops,
                             rocgraph_bool                                   drop_multi_edges,
                             rocgraph_bool                                   do_expensive_check,
                             rocgraph_graph_t**                              graph,
                             rocgraph_error_t**                              error)
{
    constexpr bool   multi_gpu = false;
    constexpr size_t int32_threshold{std::numeric_limits<int32_t>::max()};

    *graph = nullptr;
    *error = nullptr;

    auto p_handle = handle;
    auto p_vertices
        = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
            vertices);
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

    CAPI_EXPECTS(p_src->size_ == p_dst->size_,
                 rocgraph_status_invalid_input,
                 "Invalid input arguments: src size != dst size.",
                 *error);

    CAPI_EXPECTS((p_vertices == nullptr) || (p_src->type_ == p_vertices->type_),
                 rocgraph_status_invalid_input,
                 "Invalid input arguments: src type != vertices type.",
                 *error);

    CAPI_EXPECTS(p_src->type_ == p_dst->type_,
                 rocgraph_status_invalid_input,
                 "Invalid input arguments: src type != dst type.",
                 *error);

    CAPI_EXPECTS((weights == nullptr) || (p_weights->size_ == p_src->size_),
                 rocgraph_status_invalid_input,
                 "Invalid input arguments: src size != weights size.",
                 *error);

    rocgraph_data_type_id edge_type;
    rocgraph_data_type_id weight_type;

    if(p_src->size_ < int32_threshold)
    {
        edge_type = p_src->type_;
    }
    else
    {
        edge_type = rocgraph_data_type_id_int64;
    }

    if(weights != nullptr)
    {
        weight_type = p_weights->type_;
    }
    else
    {
        weight_type = rocgraph_data_type_id_float32;
    }

    CAPI_EXPECTS((edge_ids == nullptr) || (p_edge_ids->type_ == edge_type),
                 rocgraph_status_invalid_input,
                 "Invalid input arguments: Edge id type must match edge type",
                 *error);

    CAPI_EXPECTS((edge_ids == nullptr) || (p_edge_ids->size_ == p_src->size_),
                 rocgraph_status_invalid_input,
                 "Invalid input arguments: src size != edge id prop size",
                 *error);

    CAPI_EXPECTS((edge_type_ids == nullptr) || (p_edge_type_ids->size_ == p_src->size_),
                 rocgraph_status_invalid_input,
                 "Invalid input arguments: src size != edge type prop size",
                 *error);

    rocgraph_data_type_id edge_type_id_type = rocgraph_data_type_id_int32;
    if(edge_type_ids != nullptr)
    {
        edge_type_id_type = p_edge_type_ids->type_;
    }

    ::create_graph_functor functor(*p_handle->get_raft_handle(),
                                   properties,
                                   p_vertices,
                                   p_src,
                                   p_dst,
                                   p_weights,
                                   p_edge_ids,
                                   p_edge_type_ids,
                                   renumber,
                                   drop_self_loops,
                                   drop_multi_edges,
                                   do_expensive_check,
                                   edge_type);

    try
    {
        rocgraph::c_api::vertex_dispatcher(p_src->type_,
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

        *graph = reinterpret_cast<rocgraph_graph_t*>(functor.result_);
    }
    catch(std::exception const& ex)
    {
        *error
            = reinterpret_cast<rocgraph_error_t*>(new rocgraph::c_api::rocgraph_error_t{ex.what()});
        return rocgraph_status_unknown_error;
    }

    return rocgraph_status_success;
}

extern "C" rocgraph_status
    rocgraph_sg_graph_create(const rocgraph_handle_t*                        handle,
                             const rocgraph_graph_properties_t*              properties,
                             const rocgraph_type_erased_device_array_view_t* src,
                             const rocgraph_type_erased_device_array_view_t* dst,
                             const rocgraph_type_erased_device_array_view_t* weights,
                             const rocgraph_type_erased_device_array_view_t* edge_ids,
                             const rocgraph_type_erased_device_array_view_t* edge_type_ids,
                             rocgraph_bool                                   store_transposed,
                             rocgraph_bool                                   renumber,
                             rocgraph_bool                                   do_expensive_check,
                             rocgraph_graph_t**                              graph,
                             rocgraph_error_t**                              error)
{
    return rocgraph_graph_create_sg(handle,
                                    properties,
                                    NULL,
                                    src,
                                    dst,
                                    weights,
                                    edge_ids,
                                    edge_type_ids,
                                    store_transposed,
                                    renumber,
                                    rocgraph_bool_false,
                                    rocgraph_bool_false,
                                    do_expensive_check,
                                    graph,
                                    error);
}

rocgraph_status
    rocgraph_graph_create_sg_from_csr(const rocgraph_handle_t*                        handle,
                                      const rocgraph_graph_properties_t*              properties,
                                      const rocgraph_type_erased_device_array_view_t* offsets,
                                      const rocgraph_type_erased_device_array_view_t* indices,
                                      const rocgraph_type_erased_device_array_view_t* weights,
                                      const rocgraph_type_erased_device_array_view_t* edge_ids,
                                      const rocgraph_type_erased_device_array_view_t* edge_type_ids,
                                      rocgraph_bool      store_transposed,
                                      rocgraph_bool      renumber,
                                      rocgraph_bool      do_expensive_check,
                                      rocgraph_graph_t** graph,
                                      rocgraph_error_t** error)
{
    constexpr bool multi_gpu = false;

    *graph = nullptr;
    *error = nullptr;

    auto p_handle = handle;
    auto p_offsets
        = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
            offsets);
    auto p_indices
        = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
            indices);
    auto p_weights
        = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
            weights);
    auto p_edge_ids
        = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
            edge_ids);
    auto p_edge_type_ids
        = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
            edge_type_ids);

    CAPI_EXPECTS((weights == nullptr) || (p_weights->size_ == p_indices->size_),
                 rocgraph_status_invalid_input,
                 "Invalid input arguments: src size != weights size.",
                 *error);

    rocgraph_data_type_id weight_type;

    if(weights != nullptr)
    {
        weight_type = p_weights->type_;
    }
    else
    {
        weight_type = rocgraph_data_type_id_float32;
    }

    CAPI_EXPECTS(
        (edge_type_ids == nullptr && edge_ids == nullptr)
            || (edge_type_ids != nullptr && edge_ids != nullptr),
        rocgraph_status_invalid_input,
        "Invalid input arguments: either none or both of edge ids and edge types must be provided.",
        *error);

    CAPI_EXPECTS((edge_type_ids == nullptr && edge_ids == nullptr)
                     || (p_edge_ids->type_ == p_offsets->type_),
                 rocgraph_status_invalid_input,
                 "Invalid input arguments: Edge id type must match edge type",
                 *error);

    CAPI_EXPECTS((edge_type_ids == nullptr && edge_ids == nullptr)
                     || (p_edge_ids->size_ == p_indices->size_
                         && p_edge_type_ids->size_ == p_indices->size_),
                 rocgraph_status_invalid_input,
                 "Invalid input arguments: src size != edge prop size",
                 *error);

    ::create_graph_csr_functor functor(*p_handle->get_raft_handle(),
                                       properties,
                                       p_offsets,
                                       p_indices,
                                       p_weights,
                                       p_edge_ids,
                                       p_edge_type_ids,
                                       renumber,
                                       do_expensive_check);

    try
    {
        rocgraph::c_api::vertex_dispatcher(p_indices->type_,
                                           p_offsets->type_,
                                           weight_type,
                                           p_indices->type_,
                                           store_transposed,
                                           multi_gpu,
                                           functor);

        if(functor.status_ != rocgraph_status_success)
        {
            *error = reinterpret_cast<rocgraph_error_t*>(functor.error_.release());
            return functor.status_;
        }

        *graph = reinterpret_cast<rocgraph_graph_t*>(functor.result_);
    }
    catch(std::exception const& ex)
    {
        *error
            = reinterpret_cast<rocgraph_error_t*>(new rocgraph::c_api::rocgraph_error_t{ex.what()});
        return rocgraph_status_unknown_error;
    }

    return rocgraph_status_success;
}

rocgraph_status
    rocgraph_sg_graph_create_from_csr(const rocgraph_handle_t*                        handle,
                                      const rocgraph_graph_properties_t*              properties,
                                      const rocgraph_type_erased_device_array_view_t* offsets,
                                      const rocgraph_type_erased_device_array_view_t* indices,
                                      const rocgraph_type_erased_device_array_view_t* weights,
                                      const rocgraph_type_erased_device_array_view_t* edge_ids,
                                      const rocgraph_type_erased_device_array_view_t* edge_type_ids,
                                      rocgraph_bool      store_transposed,
                                      rocgraph_bool      renumber,
                                      rocgraph_bool      do_expensive_check,
                                      rocgraph_graph_t** graph,
                                      rocgraph_error_t** error)
{
    return rocgraph_graph_create_sg_from_csr(handle,
                                             properties,
                                             offsets,
                                             indices,
                                             weights,
                                             edge_ids,
                                             edge_type_ids,
                                             store_transposed,
                                             renumber,
                                             do_expensive_check,
                                             graph,
                                             error);
}

extern "C" void rocgraph_graph_free(rocgraph_graph_t* ptr_graph)
{
    if(ptr_graph != NULL)
    {
        auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(ptr_graph);

        destroy_graph_functor functor(internal_pointer->graph_,
                                      internal_pointer->number_map_,
                                      internal_pointer->edge_weights_,
                                      internal_pointer->edge_ids_,
                                      internal_pointer->edge_types_);

        rocgraph::c_api::vertex_dispatcher(internal_pointer->vertex_type_,
                                           internal_pointer->edge_type_,
                                           internal_pointer->weight_type_,
                                           internal_pointer->edge_type_id_type_,
                                           internal_pointer->store_transposed_,
                                           internal_pointer->multi_gpu_,
                                           functor);

        delete internal_pointer;
    }
}

extern "C" void rocgraph_sg_graph_free(rocgraph_graph_t* ptr_graph)
{
    rocgraph_graph_free(ptr_graph);
}
