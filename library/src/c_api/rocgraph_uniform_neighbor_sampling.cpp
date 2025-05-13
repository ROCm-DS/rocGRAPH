// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_abstract_functor.hpp"
#include "c_api/rocgraph_graph.hpp"
#include "c_api/rocgraph_handle.hpp"
#include "c_api/rocgraph_random.hpp"
#include "c_api/rocgraph_utils.hpp"

#include "internal/aux/rocgraph_sample_result_aux.h"
#include "internal/aux/rocgraph_sampling_options_aux.h"
#include "internal/aux/rocgraph_type_erased_device_array_aux.h"
#include "internal/rocgraph_algorithms.h"

#include "algorithms.hpp"
#include "detail/shuffle_wrappers.hpp"
#include "detail/utility_wrappers.hpp"
#include "sampling_functions.hpp"

#include <raft/core/handle.hpp>

namespace rocgraph
{
    namespace c_api
    {

        struct rocgraph_sampling_options_t
        {
            rocgraph_bool            with_replacement_{rocgraph_bool_false};
            rocgraph_bool            return_hops_{rocgraph_bool_false};
            prior_sources_behavior_t prior_sources_behavior_{
                rocgraph::prior_sources_behavior_t::DEFAULT};
            rocgraph_bool             dedupe_sources_{rocgraph_bool_false};
            rocgraph_bool             renumber_results_{rocgraph_bool_false};
            rocgraph_compression_type compression_type_{rocgraph_compression_type_coo};
            rocgraph_bool             compress_per_hop_{rocgraph_bool_false};
            rocgraph_bool             retain_seeds_{rocgraph_bool_false};
        };

        struct rocgraph_sample_result_t
        {
            rocgraph_type_erased_device_array_t* major_offsets_{nullptr};
            rocgraph_type_erased_device_array_t* majors_{nullptr};
            rocgraph_type_erased_device_array_t* minors_{nullptr};
            rocgraph_type_erased_device_array_t* edge_id_{nullptr};
            rocgraph_type_erased_device_array_t* edge_type_{nullptr};
            rocgraph_type_erased_device_array_t* wgt_{nullptr};
            rocgraph_type_erased_device_array_t* hop_{nullptr};
            rocgraph_type_erased_device_array_t* label_hop_offsets_{nullptr};
            rocgraph_type_erased_device_array_t* label_{nullptr};
            rocgraph_type_erased_device_array_t* renumber_map_{nullptr};
            rocgraph_type_erased_device_array_t* renumber_map_offsets_{nullptr};
        };

    } // namespace c_api
} // namespace rocgraph

namespace
{

    struct uniform_neighbor_sampling_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                                            handle_;
        rocgraph::c_api::rocgraph_graph_t*                               graph_{nullptr};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* start_vertices_{nullptr};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* start_vertex_labels_{
            nullptr};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* label_list_{nullptr};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* label_to_comm_rank_{
            nullptr};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* label_offsets_{nullptr};
        rocgraph::c_api::rocgraph_type_erased_host_array_view_t const*   fan_out_{nullptr};
        rocgraph::c_api::rocgraph_rng_state_t*                           rng_state_{nullptr};
        rocgraph::c_api::rocgraph_sampling_options_t                     options_{};
        bool                                                             do_expensive_check_{false};
        rocgraph::c_api::rocgraph_sample_result_t*                       result_{nullptr};

        uniform_neighbor_sampling_functor(
            rocgraph_handle_t const*                        handle,
            rocgraph_graph_t*                               graph,
            rocgraph_type_erased_device_array_view_t const* start_vertices,
            rocgraph_type_erased_device_array_view_t const* start_vertex_labels,
            rocgraph_type_erased_device_array_view_t const* label_list,
            rocgraph_type_erased_device_array_view_t const* label_to_comm_rank,
            rocgraph_type_erased_device_array_view_t const* label_offsets,
            rocgraph_type_erased_host_array_view_t const*   fan_out,
            rocgraph_rng_state_t*                           rng_state,
            rocgraph::c_api::rocgraph_sampling_options_t    options,
            bool                                            do_expensive_check)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , start_vertices_(reinterpret_cast<
                              rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                  start_vertices))
            , start_vertex_labels_(
                  reinterpret_cast<
                      rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                      start_vertex_labels))
            , label_list_(
                  reinterpret_cast<
                      rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(label_list))
            , label_to_comm_rank_(reinterpret_cast<
                                  rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                  label_to_comm_rank))
            , label_offsets_(reinterpret_cast<
                             rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                  label_offsets))
            , fan_out_(
                  reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_host_array_view_t const*>(
                      fan_out))
            , rng_state_(reinterpret_cast<rocgraph::c_api::rocgraph_rng_state_t*>(rng_state))
            , options_(options)
            , do_expensive_check_(do_expensive_check)
        {
        }

        template <typename vertex_t,
                  typename edge_t,
                  typename weight_t,
                  typename edge_type_t,
                  bool store_transposed,
                  bool multi_gpu>
        void operator()()
        {
            using label_t = int32_t;

            // FIXME: Think about how to handle SG vice MG
            if constexpr(!rocgraph::is_candidate<vertex_t, edge_t, weight_t>::value)
            {
                unsupported();
            }
            else
            {
                // uniform_nbr_sample expects store_transposed == false
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
                    rocgraph::graph_view_t<vertex_t, edge_t, true, multi_gpu>,
                    weight_t>*>(graph_->edge_weights_);

                auto edge_ids = reinterpret_cast<rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, true, multi_gpu>,
                    edge_t>*>(graph_->edge_ids_);

                auto edge_types = reinterpret_cast<rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, true, multi_gpu>,
                    edge_type_t>*>(graph_->edge_types_);

                auto number_map
                    = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

                rmm::device_uvector<vertex_t> start_vertices(start_vertices_->size_,
                                                             handle_.get_stream());
                raft::copy(start_vertices.data(),
                           start_vertices_->as_type<vertex_t>(),
                           start_vertices.size(),
                           handle_.get_stream());

                std::optional<rmm::device_uvector<label_t>> start_vertex_labels{std::nullopt};

                if(start_vertex_labels_ != nullptr)
                {
                    start_vertex_labels = rmm::device_uvector<label_t>{start_vertex_labels_->size_,
                                                                       handle_.get_stream()};
                    raft::copy(start_vertex_labels->data(),
                               start_vertex_labels_->as_type<label_t>(),
                               start_vertex_labels_->size_,
                               handle_.get_stream());
                }

                if constexpr(multi_gpu)
                {
                    if(start_vertex_labels)
                    {
                        std::tie(start_vertices, *start_vertex_labels) = rocgraph::detail::
                            shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
                                handle_,
                                std::move(start_vertices),
                                std::move(*start_vertex_labels));
                    }
                    else
                    {
                        start_vertices = rocgraph::detail::
                            shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
                                handle_, std::move(start_vertices));
                    }
                }

                //
                // Need to renumber start_vertices
                //
                rocgraph::renumber_local_ext_vertices<vertex_t, multi_gpu>(
                    handle_,
                    start_vertices.data(),
                    start_vertices.size(),
                    number_map->data(),
                    graph_view.local_vertex_partition_range_first(),
                    graph_view.local_vertex_partition_range_last(),
                    do_expensive_check_);

                auto&& [src, dst, wgt, edge_id, edge_type, hop, edge_label, offsets]
                    = rocgraph::uniform_neighbor_sample(
                        handle_,
                        graph_view,
                        (edge_weights != nullptr) ? std::make_optional(edge_weights->view())
                                                  : std::nullopt,
                        (edge_ids != nullptr) ? std::make_optional(edge_ids->view()) : std::nullopt,
                        (edge_types != nullptr) ? std::make_optional(edge_types->view())
                                                : std::nullopt,
                        raft::device_span<vertex_t const>{start_vertices.data(),
                                                          start_vertices.size()},
                        (start_vertex_labels_ != nullptr)
                            ? std::make_optional<raft::device_span<label_t const>>(
                                  start_vertex_labels->data(), start_vertex_labels->size())
                            : std::nullopt,
                        (label_list_ != nullptr)
                            ? std::make_optional(std::make_tuple(
                                  raft::device_span<label_t const>{label_list_->as_type<label_t>(),
                                                                   label_list_->size_},
                                  raft::device_span<label_t const>{
                                      label_to_comm_rank_->as_type<label_t>(),
                                      label_to_comm_rank_->size_}))
                            : std::nullopt,
                        raft::host_span<const int>(fan_out_->as_type<const int>(), fan_out_->size_),
                        rng_state_->rng_state_,
                        options_.return_hops_,
                        options_.with_replacement_,
                        options_.prior_sources_behavior_,
                        options_.dedupe_sources_,
                        do_expensive_check_);

                std::vector<vertex_t> vertex_partition_lasts
                    = graph_view.vertex_partition_range_lasts();

                rocgraph::unrenumber_int_vertices<vertex_t, multi_gpu>(handle_,
                                                                       src.data(),
                                                                       src.size(),
                                                                       number_map->data(),
                                                                       vertex_partition_lasts,
                                                                       do_expensive_check_);

                rocgraph::unrenumber_int_vertices<vertex_t, multi_gpu>(handle_,
                                                                       dst.data(),
                                                                       dst.size(),
                                                                       number_map->data(),
                                                                       vertex_partition_lasts,
                                                                       do_expensive_check_);

                std::optional<rmm::device_uvector<vertex_t>> majors{std::nullopt};
                rmm::device_uvector<vertex_t>                minors(0, handle_.get_stream());
                std::optional<rmm::device_uvector<size_t>>   major_offsets{std::nullopt};

                std::optional<rmm::device_uvector<size_t>> label_hop_offsets{std::nullopt};

                std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};
                std::optional<rmm::device_uvector<size_t>>   renumber_map_offsets{std::nullopt};

                bool src_is_major
                    = (options_.compression_type_ == rocgraph_compression_type_csr)
                      || (options_.compression_type_ == rocgraph_compression_type_dcsr)
                      || (options_.compression_type_ == rocgraph_compression_type_coo);

                if(options_.renumber_results_)
                {
                    if(options_.compression_type_ == rocgraph_compression_type_coo)
                    {
                        // COO

                        rmm::device_uvector<vertex_t> output_majors(0, handle_.get_stream());
                        rmm::device_uvector<vertex_t> output_renumber_map(0, handle_.get_stream());
                        std::tie(output_majors,
                                 minors,
                                 wgt,
                                 edge_id,
                                 edge_type,
                                 label_hop_offsets,
                                 output_renumber_map,
                                 renumber_map_offsets)
                            = rocgraph::renumber_and_sort_sampled_edgelist<vertex_t>(
                                handle_,
                                std::move(src),
                                std::move(dst),
                                std::move(wgt),
                                std::move(edge_id),
                                std::move(edge_type),
                                std::move(hop),
                                options_.retain_seeds_
                                    ? std::make_optional(raft::device_span<vertex_t const>{
                                          start_vertices_->as_type<vertex_t>(),
                                          start_vertices_->size_})
                                    : std::nullopt,
                                options_.retain_seeds_
                                    ? std::make_optional(raft::device_span<size_t const>{
                                          label_offsets_->as_type<size_t>(), label_offsets_->size_})
                                    : std::nullopt,
                                offsets ? std::make_optional(raft::device_span<size_t const>{
                                              offsets->data(), offsets->size()})
                                        : std::nullopt,
                                edge_label ? edge_label->size() : size_t{1},
                                hop ? fan_out_->size_ : size_t{1},
                                src_is_major,
                                do_expensive_check_);

                        majors.emplace(std::move(output_majors));
                        renumber_map.emplace(std::move(output_renumber_map));
                    }
                    else
                    {
                        // (D)CSC, (D)CSR

                        bool doubly_compress
                            = (options_.compression_type_ == rocgraph_compression_type_dcsr)
                              || (options_.compression_type_ == rocgraph_compression_type_dcsc);

                        rmm::device_uvector<size_t>   output_major_offsets(0, handle_.get_stream());
                        rmm::device_uvector<vertex_t> output_renumber_map(0, handle_.get_stream());
                        std::tie(majors,
                                 output_major_offsets,
                                 minors,
                                 wgt,
                                 edge_id,
                                 edge_type,
                                 label_hop_offsets,
                                 output_renumber_map,
                                 renumber_map_offsets)
                            = rocgraph::renumber_and_compress_sampled_edgelist<vertex_t>(
                                handle_,
                                std::move(src),
                                std::move(dst),
                                std::move(wgt),
                                std::move(edge_id),
                                std::move(edge_type),
                                std::move(hop),
                                options_.retain_seeds_
                                    ? std::make_optional(raft::device_span<vertex_t const>{
                                          start_vertices_->as_type<vertex_t>(),
                                          start_vertices_->size_})
                                    : std::nullopt,
                                options_.retain_seeds_
                                    ? std::make_optional(raft::device_span<size_t const>{
                                          label_offsets_->as_type<size_t>(), label_offsets_->size_})
                                    : std::nullopt,
                                offsets ? std::make_optional(raft::device_span<size_t const>{
                                              offsets->data(), offsets->size()})
                                        : std::nullopt,
                                edge_label ? edge_label->size() : size_t{1},
                                hop ? fan_out_->size_ : size_t{1},
                                src_is_major,
                                options_.compress_per_hop_,
                                doubly_compress,
                                do_expensive_check_);

                        renumber_map.emplace(std::move(output_renumber_map));
                        major_offsets.emplace(std::move(output_major_offsets));
                    }

                    // These are now represented by label_hop_offsets
                    hop.reset();
                    offsets.reset();
                }
                else
                {
                    if(options_.compression_type_ != rocgraph_compression_type_coo)
                    {
                        ROCGRAPH_FAIL("Can only use COO format if not renumbering");
                    }

                    std::tie(src, dst, wgt, edge_id, edge_type, label_hop_offsets)
                        = rocgraph::sort_sampled_edgelist(
                            handle_,
                            std::move(src),
                            std::move(dst),
                            std::move(wgt),
                            std::move(edge_id),
                            std::move(edge_type),
                            std::move(hop),
                            offsets ? std::make_optional(raft::device_span<size_t const>{
                                          offsets->data(), offsets->size()})
                                    : std::nullopt,
                            edge_label ? edge_label->size() : size_t{1},
                            hop ? fan_out_->size_ : size_t{1},
                            src_is_major,
                            do_expensive_check_);

                    majors.emplace(std::move(src));
                    minors = std::move(dst);

                    hop.reset();
                    offsets.reset();
                }

                result_ = new rocgraph::c_api::rocgraph_sample_result_t{
                    (major_offsets) ? new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                                          *major_offsets, rocgraph_data_type_id_size_t)
                                    : nullptr,
                    (majors) ? new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                                   *majors, graph_->vertex_type_)
                             : nullptr,
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(minors,
                                                                             graph_->vertex_type_),
                    (edge_id) ? new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                                    *edge_id, graph_->edge_type_)
                              : nullptr,
                    (edge_type) ? new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                                      *edge_type, graph_->edge_type_id_type_)
                                : nullptr,
                    (wgt) ? new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                                *wgt, graph_->weight_type_)
                          : nullptr,
                    (hop) ? new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                                *hop,
                                rocgraph_data_type_id_int32)
                          : nullptr, // FIXME get rid of this
                    (label_hop_offsets) ? new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                                              *label_hop_offsets, rocgraph_data_type_id_size_t)
                                        : nullptr,
                    (edge_label) ? new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                                       edge_label.value(), rocgraph_data_type_id_int32)
                                 : nullptr,
                    (renumber_map) ? new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                                         renumber_map.value(), graph_->vertex_type_)
                                   : nullptr,
                    (renumber_map_offsets)
                        ? new rocgraph::c_api::rocgraph_type_erased_device_array_t(
                              renumber_map_offsets.value(), rocgraph_data_type_id_size_t)
                        : nullptr};
            }
        }
    };

} // namespace

extern "C" rocgraph_status rocgraph_sampling_options_create(rocgraph_sampling_options_t** options,
                                                            rocgraph_error_t**            error)
{
    *options = reinterpret_cast<rocgraph_sampling_options_t*>(
        new rocgraph::c_api::rocgraph_sampling_options_t());
    if(*options == nullptr)
    {
        *error = reinterpret_cast<rocgraph_error_t*>(
            new rocgraph::c_api::rocgraph_error_t{"invalid resource handle"});
        return rocgraph_status_invalid_handle;
    }

    return rocgraph_status_success;
}

extern "C" void rocgraph_sampling_set_retain_seeds(rocgraph_sampling_options_t* options,
                                                   rocgraph_bool                value)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_sampling_options_t*>(options);
    internal_pointer->retain_seeds_ = value;
}

extern "C" void rocgraph_sampling_set_renumber_results(rocgraph_sampling_options_t* options,
                                                       rocgraph_bool                value)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_sampling_options_t*>(options);
    internal_pointer->renumber_results_ = value;
}

extern "C" void rocgraph_sampling_set_compress_per_hop(rocgraph_sampling_options_t* options,
                                                       rocgraph_bool                value)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_sampling_options_t*>(options);
    internal_pointer->compress_per_hop_ = value;
}

extern "C" void rocgraph_sampling_set_with_replacement(rocgraph_sampling_options_t* options,
                                                       rocgraph_bool                value)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_sampling_options_t*>(options);
    internal_pointer->with_replacement_ = value;
}

extern "C" void rocgraph_sampling_set_return_hops(rocgraph_sampling_options_t* options,
                                                  rocgraph_bool                value)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_sampling_options_t*>(options);
    internal_pointer->return_hops_ = value;
}

extern "C" void rocgraph_sampling_set_compression_type(rocgraph_sampling_options_t* options,
                                                       rocgraph_compression_type    value)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_sampling_options_t*>(options);
    switch(value)
    {
    case rocgraph_compression_type_coo:
        internal_pointer->compression_type_ = rocgraph_compression_type_coo;
        break;
    case rocgraph_compression_type_csr:
        internal_pointer->compression_type_ = rocgraph_compression_type_csr;
        break;
    case rocgraph_compression_type_csc:
        internal_pointer->compression_type_ = rocgraph_compression_type_csc;
        break;
    case rocgraph_compression_type_dcsr:
        internal_pointer->compression_type_ = rocgraph_compression_type_dcsr;
        break;
    case rocgraph_compression_type_dcsc:
        internal_pointer->compression_type_ = rocgraph_compression_type_dcsc;
        break;
    default:
        ROCGRAPH_FAIL("Invalid compression type");
    }
}

extern "C" void rocgraph_sampling_set_prior_sources_behavior(rocgraph_sampling_options_t* options,
                                                             rocgraph_prior_sources_behavior value)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_sampling_options_t*>(options);
    switch(value)
    {
    case rocgraph_prior_sources_behavior_carry_over:
        internal_pointer->prior_sources_behavior_ = rocgraph::prior_sources_behavior_t::CARRY_OVER;
        break;
    case rocgraph_prior_sources_behavior_exclude:
        internal_pointer->prior_sources_behavior_ = rocgraph::prior_sources_behavior_t::EXCLUDE;
        break;
    default:
        internal_pointer->prior_sources_behavior_ = rocgraph::prior_sources_behavior_t::DEFAULT;
        break;
    }
}

extern "C" void rocgraph_sampling_set_dedupe_sources(rocgraph_sampling_options_t* options,
                                                     rocgraph_bool                value)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_sampling_options_t*>(options);
    internal_pointer->dedupe_sources_ = value;
}

extern "C" void rocgraph_sampling_options_free(rocgraph_sampling_options_t* options)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_sampling_options_t*>(options);
    delete internal_pointer;
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_sources(const rocgraph_sample_result_t* result)
{
    // Deprecated.
    return rocgraph_sample_result_get_majors(result);
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_destinations(const rocgraph_sample_result_t* result)
{
    // Deprecated.
    return rocgraph_sample_result_get_minors(result);
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_majors(const rocgraph_sample_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_sample_result_t const*>(result);
    return (internal_pointer->majors_ != nullptr)
               ? reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
                     internal_pointer->majors_->view())

               : NULL;
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_major_offsets(const rocgraph_sample_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_sample_result_t const*>(result);
    return (internal_pointer->major_offsets_ != nullptr)
               ? reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
                     internal_pointer->major_offsets_->view())

               : NULL;
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_minors(const rocgraph_sample_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_sample_result_t const*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->minors_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_start_labels(const rocgraph_sample_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_sample_result_t const*>(result);
    return internal_pointer->label_ != nullptr
               ? reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
                     internal_pointer->label_->view())
               : NULL;
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_edge_id(const rocgraph_sample_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_sample_result_t const*>(result);
    return internal_pointer->edge_id_ != nullptr
               ? reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
                     internal_pointer->edge_id_->view())
               : NULL;
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_edge_type(const rocgraph_sample_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_sample_result_t const*>(result);
    return internal_pointer->edge_type_ != nullptr
               ? reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
                     internal_pointer->edge_type_->view())
               : NULL;
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_edge_weight(const rocgraph_sample_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_sample_result_t const*>(result);
    return internal_pointer->wgt_ != nullptr
               ? reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
                     internal_pointer->wgt_->view())
               : NULL;
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_hop(const rocgraph_sample_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_sample_result_t const*>(result);
    return internal_pointer->hop_ != nullptr
               ? reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
                     internal_pointer->hop_->view())
               : NULL;
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_label_hop_offsets(const rocgraph_sample_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_sample_result_t const*>(result);
    return internal_pointer->label_hop_offsets_ != nullptr
               ? reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
                     internal_pointer->label_hop_offsets_->view())
               : NULL;
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_index(const rocgraph_sample_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_sample_result_t const*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->edge_id_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_offsets(const rocgraph_sample_result_t* result)
{
    // Deprecated.
    return rocgraph_sample_result_get_label_hop_offsets(result);
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_renumber_map(const rocgraph_sample_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_sample_result_t const*>(result);
    return internal_pointer->renumber_map_ == nullptr
               ? NULL
               : reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
                     internal_pointer->renumber_map_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_renumber_map_offsets(const rocgraph_sample_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_sample_result_t const*>(result);
    return internal_pointer->renumber_map_ == nullptr
               ? NULL
               : reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
                     internal_pointer->renumber_map_offsets_->view());
}

extern "C" rocgraph_status rocgraph_test_uniform_neighborhood_sample_result_create(
    const rocgraph_handle_t*                        handle,
    const rocgraph_type_erased_device_array_view_t* srcs,
    const rocgraph_type_erased_device_array_view_t* dsts,
    const rocgraph_type_erased_device_array_view_t* edge_id,
    const rocgraph_type_erased_device_array_view_t* edge_type,
    const rocgraph_type_erased_device_array_view_t* weight,
    const rocgraph_type_erased_device_array_view_t* hop,
    const rocgraph_type_erased_device_array_view_t* label,
    rocgraph_sample_result_t**                      result,
    rocgraph_error_t**                              error)
{
    *result = nullptr;
    *error  = nullptr;
    size_t          n_bytes{0};
    rocgraph_status status{rocgraph_status_success};

    if(!handle)
    {
        *error = reinterpret_cast<rocgraph_error_t*>(
            new rocgraph::c_api::rocgraph_error_t{"invalid resource handle"});
        return rocgraph_status_invalid_handle;
    }

    // Create unique_ptrs and release them during rocgraph_sample_result_t
    // construction. This allows the arrays to be cleaned up if this function
    // returns early on error.
    using device_array_unique_ptr_t
        = std::unique_ptr<rocgraph_type_erased_device_array_t,
                          decltype(&rocgraph_type_erased_device_array_free)>;

    // copy srcs to new device array
    rocgraph_type_erased_device_array_t* new_device_srcs_ptr{nullptr};
    status = rocgraph_type_erased_device_array_create_from_view(
        handle, srcs, &new_device_srcs_ptr, error);
    if(status != rocgraph_status_success)
        return status;

    device_array_unique_ptr_t new_device_srcs(new_device_srcs_ptr,
                                              &rocgraph_type_erased_device_array_free);

    // copy dsts to new device array
    rocgraph_type_erased_device_array_t* new_device_dsts_ptr{nullptr};
    status = rocgraph_type_erased_device_array_create_from_view(
        handle, dsts, &new_device_dsts_ptr, error);
    if(status != rocgraph_status_success)
        return status;

    device_array_unique_ptr_t new_device_dsts(new_device_dsts_ptr,
                                              &rocgraph_type_erased_device_array_free);

    // copy weights to new device array
    rocgraph_type_erased_device_array_t* new_device_weight_ptr{nullptr};
    status = rocgraph_type_erased_device_array_create_from_view(
        handle, weight, &new_device_weight_ptr, error);
    if(status != rocgraph_status_success)
        return status;

    device_array_unique_ptr_t new_device_weight(new_device_weight_ptr,
                                                &rocgraph_type_erased_device_array_free);

    // copy edge ids to new device array
    rocgraph_type_erased_device_array_t* new_device_edge_id_ptr{nullptr};
    status = rocgraph_type_erased_device_array_create_from_view(
        handle, edge_id, &new_device_edge_id_ptr, error);
    if(status != rocgraph_status_success)
        return status;

    device_array_unique_ptr_t new_device_edge_id(new_device_edge_id_ptr,
                                                 &rocgraph_type_erased_device_array_free);

    // copy edge types to new device array
    rocgraph_type_erased_device_array_t* new_device_edge_type_ptr{nullptr};
    status = rocgraph_type_erased_device_array_create_from_view(
        handle, edge_type, &new_device_edge_type_ptr, error);
    if(status != rocgraph_status_success)
        return status;

    device_array_unique_ptr_t new_device_edge_type(new_device_edge_type_ptr,
                                                   &rocgraph_type_erased_device_array_free);
    // copy hop ids to new device array
    rocgraph_type_erased_device_array_t* new_device_hop_ptr{nullptr};
    status = rocgraph_type_erased_device_array_create_from_view(
        handle, hop, &new_device_hop_ptr, error);
    if(status != rocgraph_status_success)
        return status;

    device_array_unique_ptr_t new_device_hop(new_device_hop_ptr,
                                             &rocgraph_type_erased_device_array_free);

    // copy labels to new device array
    rocgraph_type_erased_device_array_t* new_device_label_ptr{nullptr};
    status = rocgraph_type_erased_device_array_create_from_view(
        handle, label, &new_device_label_ptr, error);
    if(status != rocgraph_status_success)
        return status;

    device_array_unique_ptr_t new_device_label(new_device_label_ptr,
                                               &rocgraph_type_erased_device_array_free);

    // create new rocgraph_sample_result_t
    *result
        = reinterpret_cast<rocgraph_sample_result_t*>(new rocgraph::c_api::rocgraph_sample_result_t{
            nullptr,
            reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_t*>(
                new_device_srcs.release()),
            reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_t*>(
                new_device_dsts.release()),
            reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_t*>(
                new_device_edge_id.release()),
            reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_t*>(
                new_device_edge_type.release()),
            reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_t*>(
                new_device_weight.release()),
            reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_t*>(
                new_device_hop.release()),
            reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_t*>(
                new_device_label.release())});

    return rocgraph_status_success;
}

extern "C" rocgraph_status
    rocgraph_test_sample_result_create(const rocgraph_handle_t*                        handle,
                                       const rocgraph_type_erased_device_array_view_t* srcs,
                                       const rocgraph_type_erased_device_array_view_t* dsts,
                                       const rocgraph_type_erased_device_array_view_t* edge_id,
                                       const rocgraph_type_erased_device_array_view_t* edge_type,
                                       const rocgraph_type_erased_device_array_view_t* wgt,
                                       const rocgraph_type_erased_device_array_view_t* hop,
                                       const rocgraph_type_erased_device_array_view_t* label,
                                       rocgraph_sample_result_t**                      result,
                                       rocgraph_error_t**                              error)
{
    *result = nullptr;
    *error  = nullptr;
    size_t          n_bytes{0};
    rocgraph_status status{rocgraph_status_success};

    if(!handle)
    {
        *error = reinterpret_cast<rocgraph_error_t*>(
            new rocgraph::c_api::rocgraph_error_t{"invalid resource handle"});
        return rocgraph_status_invalid_handle;
    }

    // Create unique_ptrs and release them during rocgraph_sample_result_t
    // construction. This allows the arrays to be cleaned up if this function
    // returns early on error.
    using device_array_unique_ptr_t
        = std::unique_ptr<rocgraph_type_erased_device_array_t,
                          decltype(&rocgraph_type_erased_device_array_free)>;

    // copy srcs to new device array
    rocgraph_type_erased_device_array_t* new_device_srcs_ptr{nullptr};
    status = rocgraph_type_erased_device_array_create_from_view(
        handle, srcs, &new_device_srcs_ptr, error);
    if(status != rocgraph_status_success)
        return status;

    device_array_unique_ptr_t new_device_srcs(new_device_srcs_ptr,
                                              &rocgraph_type_erased_device_array_free);

    // copy dsts to new device array
    rocgraph_type_erased_device_array_t* new_device_dsts_ptr{nullptr};
    status = rocgraph_type_erased_device_array_create_from_view(
        handle, dsts, &new_device_dsts_ptr, error);
    if(status != rocgraph_status_success)
        return status;

    device_array_unique_ptr_t new_device_dsts(new_device_dsts_ptr,
                                              &rocgraph_type_erased_device_array_free);

    // copy edge_id to new device array
    rocgraph_type_erased_device_array_t* new_device_edge_id_ptr{nullptr};

    if(edge_id != NULL)
    {
        status = rocgraph_type_erased_device_array_create_from_view(
            handle, edge_id, &new_device_edge_id_ptr, error);
        if(status != rocgraph_status_success)
            return status;
    }

    device_array_unique_ptr_t new_device_edge_id(new_device_edge_id_ptr,
                                                 &rocgraph_type_erased_device_array_free);

    // copy edge_type to new device array
    rocgraph_type_erased_device_array_t* new_device_edge_type_ptr{nullptr};

    if(edge_type != NULL)
    {
        status = rocgraph_type_erased_device_array_create_from_view(
            handle, edge_type, &new_device_edge_type_ptr, error);
        if(status != rocgraph_status_success)
            return status;
    }

    device_array_unique_ptr_t new_device_edge_type(new_device_edge_type_ptr,
                                                   &rocgraph_type_erased_device_array_free);

    // copy wgt to new device array
    rocgraph_type_erased_device_array_t* new_device_wgt_ptr{nullptr};
    if(wgt != NULL)
    {
        status = rocgraph_type_erased_device_array_create_from_view(
            handle, wgt, &new_device_wgt_ptr, error);
        if(status != rocgraph_status_success)
            return status;
    }

    device_array_unique_ptr_t new_device_wgt(new_device_wgt_ptr,
                                             &rocgraph_type_erased_device_array_free);

    // copy hop to new device array
    rocgraph_type_erased_device_array_t* new_device_hop_ptr{nullptr};
    status = rocgraph_type_erased_device_array_create_from_view(
        handle, hop, &new_device_hop_ptr, error);
    if(status != rocgraph_status_success)
        return status;

    device_array_unique_ptr_t new_device_hop(new_device_hop_ptr,
                                             &rocgraph_type_erased_device_array_free);

    // copy label to new device array
    rocgraph_type_erased_device_array_t* new_device_label_ptr{nullptr};

    if(label != NULL)
    {
        status = rocgraph_type_erased_device_array_create_from_view(
            handle, label, &new_device_label_ptr, error);
        if(status != rocgraph_status_success)
            return status;
    }

    device_array_unique_ptr_t new_device_label(new_device_label_ptr,
                                               &rocgraph_type_erased_device_array_free);

    // create new rocgraph_sample_result_t
    *result
        = reinterpret_cast<rocgraph_sample_result_t*>(new rocgraph::c_api::rocgraph_sample_result_t{
            reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_t*>(
                new_device_srcs.release()),
            reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_t*>(
                new_device_dsts.release()),
            reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_t*>(
                new_device_edge_id.release()),
            reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_t*>(
                new_device_edge_type.release()),
            reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_t*>(
                new_device_wgt.release()),
            reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_t*>(
                new_device_label.release()),
            reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_t*>(
                new_device_hop.release())});

    return rocgraph_status_success;
}

extern "C" void rocgraph_sample_result_free(rocgraph_sample_result_t* result)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_sample_result_t*>(result);
    delete internal_pointer->major_offsets_;
    delete internal_pointer->majors_;
    delete internal_pointer->minors_;
    delete internal_pointer->edge_id_;
    delete internal_pointer->edge_type_;
    delete internal_pointer->wgt_;
    delete internal_pointer->hop_;
    delete internal_pointer->label_hop_offsets_;
    delete internal_pointer->label_;
    delete internal_pointer->renumber_map_;
    delete internal_pointer->renumber_map_offsets_;
    delete internal_pointer;
}

rocgraph_status rocgraph_uniform_neighbor_sample(
    const rocgraph_handle_t*                        handle,
    rocgraph_graph_t*                               graph,
    const rocgraph_type_erased_device_array_view_t* start_vertices,
    const rocgraph_type_erased_device_array_view_t* start_vertex_labels,
    const rocgraph_type_erased_device_array_view_t* label_list,
    const rocgraph_type_erased_device_array_view_t* label_to_comm_rank,
    const rocgraph_type_erased_device_array_view_t* label_offsets,
    const rocgraph_type_erased_host_array_view_t*   fan_out,
    rocgraph_rng_state_t*                           rng_state,
    const rocgraph_sampling_options_t*              options,
    rocgraph_bool                                   do_expensive_check,
    rocgraph_sample_result_t**                      result,
    rocgraph_error_t**                              error)
{
    auto options_cpp
        = *reinterpret_cast<rocgraph::c_api::rocgraph_sampling_options_t const*>(options);

    CAPI_EXPECTS((!options_cpp.retain_seeds_) || (label_offsets != nullptr),
                 rocgraph_status_invalid_input,
                 "must specify label_offsets if retain_seeds is true",
                 *error);

    CAPI_EXPECTS(
        (start_vertex_labels == nullptr)
            || (reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                    start_vertex_labels)
                    ->type_
                == rocgraph_data_type_id_int32),
        rocgraph_status_invalid_input,
        "start_vertex_labels should be of type int",
        *error);

    CAPI_EXPECTS((label_to_comm_rank == nullptr) || (start_vertex_labels != nullptr),
                 rocgraph_status_invalid_input,
                 "cannot specify label_to_comm_rank unless start_vertex_labels is also specified",
                 *error);

    CAPI_EXPECTS((label_to_comm_rank == nullptr) || (label_list != nullptr),
                 rocgraph_status_invalid_input,
                 "cannot specify label_to_comm_rank unless label_list is also specified",
                 *error);

    CAPI_EXPECTS(
        reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph)->vertex_type_
            == reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                   start_vertices)
                   ->type_,
        rocgraph_status_invalid_input,
        "vertex type of graph and start_vertices must match",
        *error);

    CAPI_EXPECTS(
        reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_host_array_view_t const*>(fan_out)
                ->type_
            == rocgraph_data_type_id_int32,
        rocgraph_status_invalid_input,
        "fan_out should be of type int",
        *error);

    uniform_neighbor_sampling_functor functor{handle,
                                              graph,
                                              start_vertices,
                                              start_vertex_labels,
                                              label_list,
                                              label_to_comm_rank,
                                              label_offsets,
                                              fan_out,
                                              rng_state,
                                              std::move(options_cpp),
                                              do_expensive_check};
    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}
