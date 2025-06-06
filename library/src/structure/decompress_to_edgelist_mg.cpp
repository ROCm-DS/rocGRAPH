// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "structure/decompress_to_edgelist_impl.cuh"

namespace rocgraph
{

    // MG instantiation

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<float>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<int32_t>>>
        decompress_to_edgelist<int32_t, int32_t, float, int32_t, false, true>(
            raft::handle_t const&                                        handle,
            graph_view_t<int32_t, int32_t, false, true> const&           graph_view,
            std::optional<edge_property_view_t<int32_t, float const*>>   edge_weight_view,
            std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
            std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_type_view,
            std::optional<raft::device_span<int32_t const>>              renumber_map,
            bool                                                         do_expensive_check);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<float>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<int32_t>>>
        decompress_to_edgelist<int32_t, int32_t, float, int32_t, true, true>(
            raft::handle_t const&                                        handle,
            graph_view_t<int32_t, int32_t, true, true> const&            graph_view,
            std::optional<edge_property_view_t<int32_t, float const*>>   edge_weight_view,
            std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
            std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_type_view,
            std::optional<raft::device_span<int32_t const>>              renumber_map,
            bool                                                         do_expensive_check);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<double>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<int32_t>>>
        decompress_to_edgelist<int32_t, int32_t, double, int32_t, false, true>(
            raft::handle_t const&                                        handle,
            graph_view_t<int32_t, int32_t, false, true> const&           graph_view,
            std::optional<edge_property_view_t<int32_t, double const*>>  edge_weight_view,
            std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
            std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_type_view,
            std::optional<raft::device_span<int32_t const>>              renumber_map,
            bool                                                         do_expensive_check);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<double>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<int32_t>>>
        decompress_to_edgelist<int32_t, int32_t, double, int32_t, true, true>(
            raft::handle_t const&                                        handle,
            graph_view_t<int32_t, int32_t, true, true> const&            graph_view,
            std::optional<edge_property_view_t<int32_t, double const*>>  edge_weight_view,
            std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
            std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_type_view,
            std::optional<raft::device_span<int32_t const>>              renumber_map,
            bool                                                         do_expensive_check);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<float>>,
                        std::optional<rmm::device_uvector<int64_t>>,
                        std::optional<rmm::device_uvector<int32_t>>>
        decompress_to_edgelist<int32_t, int64_t, float, int32_t, false, true>(
            raft::handle_t const&                                        handle,
            graph_view_t<int32_t, int64_t, false, true> const&           graph_view,
            std::optional<edge_property_view_t<int64_t, float const*>>   edge_weight_view,
            std::optional<edge_property_view_t<int64_t, int64_t const*>> edge_id_view,
            std::optional<edge_property_view_t<int64_t, int32_t const*>> edge_type_view,
            std::optional<raft::device_span<int32_t const>>              renumber_map,
            bool                                                         do_expensive_check);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<float>>,
                        std::optional<rmm::device_uvector<int64_t>>,
                        std::optional<rmm::device_uvector<int32_t>>>
        decompress_to_edgelist<int32_t, int64_t, float, int32_t, true, true>(
            raft::handle_t const&                                        handle,
            graph_view_t<int32_t, int64_t, true, true> const&            graph_view,
            std::optional<edge_property_view_t<int64_t, float const*>>   edge_weight_view,
            std::optional<edge_property_view_t<int64_t, int64_t const*>> edge_id_view,
            std::optional<edge_property_view_t<int64_t, int32_t const*>> edge_type_view,
            std::optional<raft::device_span<int32_t const>>              renumber_map,
            bool                                                         do_expensive_check);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<double>>,
                        std::optional<rmm::device_uvector<int64_t>>,
                        std::optional<rmm::device_uvector<int32_t>>>
        decompress_to_edgelist<int32_t, int64_t, double, int32_t, false, true>(
            raft::handle_t const&                                        handle,
            graph_view_t<int32_t, int64_t, false, true> const&           graph_view,
            std::optional<edge_property_view_t<int64_t, double const*>>  edge_weight_view,
            std::optional<edge_property_view_t<int64_t, int64_t const*>> edge_id_view,
            std::optional<edge_property_view_t<int64_t, int32_t const*>> edge_type_view,
            std::optional<raft::device_span<int32_t const>>              renumber_map,
            bool                                                         do_expensive_check);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<double>>,
                        std::optional<rmm::device_uvector<int64_t>>,
                        std::optional<rmm::device_uvector<int32_t>>>
        decompress_to_edgelist<int32_t, int64_t, double, int32_t, true, true>(
            raft::handle_t const&                                        handle,
            graph_view_t<int32_t, int64_t, true, true> const&            graph_view,
            std::optional<edge_property_view_t<int64_t, double const*>>  edge_weight_view,
            std::optional<edge_property_view_t<int64_t, int64_t const*>> edge_id_view,
            std::optional<edge_property_view_t<int64_t, int32_t const*>> edge_type_view,
            std::optional<raft::device_span<int32_t const>>              renumber_map,
            bool                                                         do_expensive_check);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>,
                        std::optional<rmm::device_uvector<float>>,
                        std::optional<rmm::device_uvector<int64_t>>,
                        std::optional<rmm::device_uvector<int32_t>>>
        decompress_to_edgelist<int64_t, int64_t, float, int32_t, false, true>(
            raft::handle_t const&                                        handle,
            graph_view_t<int64_t, int64_t, false, true> const&           graph_view,
            std::optional<edge_property_view_t<int64_t, float const*>>   edge_weight_view,
            std::optional<edge_property_view_t<int64_t, int64_t const*>> edge_id_view,
            std::optional<edge_property_view_t<int64_t, int32_t const*>> edge_type_view,
            std::optional<raft::device_span<int64_t const>>              renumber_map,
            bool                                                         do_expensive_check);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>,
                        std::optional<rmm::device_uvector<float>>,
                        std::optional<rmm::device_uvector<int64_t>>,
                        std::optional<rmm::device_uvector<int32_t>>>
        decompress_to_edgelist<int64_t, int64_t, float, int32_t, true, true>(
            raft::handle_t const&                                        handle,
            graph_view_t<int64_t, int64_t, true, true> const&            graph_view,
            std::optional<edge_property_view_t<int64_t, float const*>>   edge_weight_view,
            std::optional<edge_property_view_t<int64_t, int64_t const*>> edge_id_view,
            std::optional<edge_property_view_t<int64_t, int32_t const*>> edge_type_view,
            std::optional<raft::device_span<int64_t const>>              renumber_map,
            bool                                                         do_expensive_check);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>,
                        std::optional<rmm::device_uvector<double>>,
                        std::optional<rmm::device_uvector<int64_t>>,
                        std::optional<rmm::device_uvector<int32_t>>>
        decompress_to_edgelist<int64_t, int64_t, double, int32_t, false, true>(
            raft::handle_t const&                                        handle,
            graph_view_t<int64_t, int64_t, false, true> const&           graph_view,
            std::optional<edge_property_view_t<int64_t, double const*>>  edge_weight_view,
            std::optional<edge_property_view_t<int64_t, int64_t const*>> edge_id_view,
            std::optional<edge_property_view_t<int64_t, int32_t const*>> edge_type_view,
            std::optional<raft::device_span<int64_t const>>              renumber_map,
            bool                                                         do_expensive_check);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>,
                        std::optional<rmm::device_uvector<double>>,
                        std::optional<rmm::device_uvector<int64_t>>,
                        std::optional<rmm::device_uvector<int32_t>>>
        decompress_to_edgelist<int64_t, int64_t, double, int32_t, true, true>(
            raft::handle_t const&                                        handle,
            graph_view_t<int64_t, int64_t, true, true> const&            graph_view,
            std::optional<edge_property_view_t<int64_t, double const*>>  edge_weight_view,
            std::optional<edge_property_view_t<int64_t, int64_t const*>> edge_id_view,
            std::optional<edge_property_view_t<int64_t, int32_t const*>> edge_type_view,
            std::optional<raft::device_span<int64_t const>>              renumber_map,
            bool                                                         do_expensive_check);

} // namespace rocgraph
