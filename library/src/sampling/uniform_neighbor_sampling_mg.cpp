// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "uniform_neighbor_sampling_impl.hpp"

#include "algorithms.hpp"

namespace rocgraph
{

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<float>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<size_t>>>
        uniform_neighbor_sample(
            raft::handle_t const&                                        handle,
            graph_view_t<int32_t, int32_t, false, true> const&           graph_view,
            std::optional<edge_property_view_t<int32_t, float const*>>   edge_weight_view,
            std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
            std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_type_view,
            raft::device_span<int32_t const>                             starting_vertices,
            std::optional<raft::device_span<int32_t const>>              starting_vertex_labels,
            std::optional<std::tuple<raft::device_span<int32_t const>,
                                     raft::device_span<int32_t const>>>  label_to_output_comm_rank,
            raft::host_span<int32_t const>                               fan_out,
            raft::random::RngState&                                      rng_state,
            bool                                                         return_hops,
            bool                                                         with_replacement,
            prior_sources_behavior_t                                     prior_sources_behavior,
            bool                                                         dedupe_sources,
            bool                                                         do_expensive_check);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<float>>,
                        std::optional<rmm::device_uvector<int64_t>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<size_t>>>
        uniform_neighbor_sample(
            raft::handle_t const&                                        handle,
            graph_view_t<int32_t, int64_t, false, true> const&           graph_view,
            std::optional<edge_property_view_t<int64_t, float const*>>   edge_weight_view,
            std::optional<edge_property_view_t<int64_t, int64_t const*>> edge_id_view,
            std::optional<edge_property_view_t<int64_t, int32_t const*>> edge_type_view,
            raft::device_span<int32_t const>                             starting_vertices,
            std::optional<raft::device_span<int32_t const>>              starting_vertex_labels,
            std::optional<std::tuple<raft::device_span<int32_t const>,
                                     raft::device_span<int32_t const>>>  label_to_output_comm_rank,
            raft::host_span<int32_t const>                               fan_out,
            raft::random::RngState&                                      rng_state,
            bool                                                         return_hops,
            bool                                                         with_replacement,
            prior_sources_behavior_t                                     prior_sources_behavior,
            bool                                                         dedupe_sources,
            bool                                                         do_expensive_check);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>,
                        std::optional<rmm::device_uvector<float>>,
                        std::optional<rmm::device_uvector<int64_t>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<size_t>>>
        uniform_neighbor_sample(
            raft::handle_t const&                                        handle,
            graph_view_t<int64_t, int64_t, false, true> const&           graph_view,
            std::optional<edge_property_view_t<int64_t, float const*>>   edge_weight_view,
            std::optional<edge_property_view_t<int64_t, int64_t const*>> edge_id_view,
            std::optional<edge_property_view_t<int64_t, int32_t const*>> edge_type_view,
            raft::device_span<int64_t const>                             starting_vertices,
            std::optional<raft::device_span<int32_t const>>              starting_vertex_labels,
            std::optional<std::tuple<raft::device_span<int32_t const>,
                                     raft::device_span<int32_t const>>>  label_to_output_comm_rank,
            raft::host_span<int32_t const>                               fan_out,
            raft::random::RngState&                                      rng_state,
            bool                                                         return_hops,
            bool                                                         with_replacement,
            prior_sources_behavior_t                                     prior_sources_behavior,
            bool                                                         dedupe_sources,
            bool                                                         do_expensive_check);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<double>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<size_t>>>
        uniform_neighbor_sample(
            raft::handle_t const&                                        handle,
            graph_view_t<int32_t, int32_t, false, true> const&           graph_view,
            std::optional<edge_property_view_t<int32_t, double const*>>  edge_weight_view,
            std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
            std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_type_view,
            raft::device_span<int32_t const>                             starting_vertices,
            std::optional<raft::device_span<int32_t const>>              starting_vertex_labels,
            std::optional<std::tuple<raft::device_span<int32_t const>,
                                     raft::device_span<int32_t const>>>  label_to_output_comm_rank,
            raft::host_span<int32_t const>                               fan_out,
            raft::random::RngState&                                      rng_state,
            bool                                                         return_hops,
            bool                                                         with_replacement,
            prior_sources_behavior_t                                     prior_sources_behavior,
            bool                                                         dedupe_sources,
            bool                                                         do_expensive_check);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<double>>,
                        std::optional<rmm::device_uvector<int64_t>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<size_t>>>
        uniform_neighbor_sample(
            raft::handle_t const&                                        handle,
            graph_view_t<int32_t, int64_t, false, true> const&           graph_view,
            std::optional<edge_property_view_t<int64_t, double const*>>  edge_weight_view,
            std::optional<edge_property_view_t<int64_t, int64_t const*>> edge_id_view,
            std::optional<edge_property_view_t<int64_t, int32_t const*>> edge_type_view,
            raft::device_span<int32_t const>                             starting_vertices,
            std::optional<raft::device_span<int32_t const>>              starting_vertex_labels,
            std::optional<std::tuple<raft::device_span<int32_t const>,
                                     raft::device_span<int32_t const>>>  label_to_output_comm_rank,
            raft::host_span<int32_t const>                               fan_out,
            raft::random::RngState&                                      rng_state,
            bool                                                         return_hops,
            bool                                                         with_replacement,
            prior_sources_behavior_t                                     prior_sources_behavior,
            bool                                                         dedupe_sources,
            bool                                                         do_expensive_check);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>,
                        std::optional<rmm::device_uvector<double>>,
                        std::optional<rmm::device_uvector<int64_t>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<size_t>>>
        uniform_neighbor_sample(
            raft::handle_t const&                                        handle,
            graph_view_t<int64_t, int64_t, false, true> const&           graph_view,
            std::optional<edge_property_view_t<int64_t, double const*>>  edge_weight_view,
            std::optional<edge_property_view_t<int64_t, int64_t const*>> edge_id_view,
            std::optional<edge_property_view_t<int64_t, int32_t const*>> edge_type_view,
            raft::device_span<int64_t const>                             starting_vertices,
            std::optional<raft::device_span<int32_t const>>              starting_vertex_labels,
            std::optional<std::tuple<raft::device_span<int32_t const>,
                                     raft::device_span<int32_t const>>>  label_to_output_comm_rank,
            raft::host_span<int32_t const>                               fan_out,
            raft::random::RngState&                                      rng_state,
            bool                                                         return_hops,
            bool                                                         with_replacement,
            prior_sources_behavior_t                                     prior_sources_behavior,
            bool                                                         dedupe_sources,
            bool                                                         do_expensive_check);

} // namespace rocgraph
