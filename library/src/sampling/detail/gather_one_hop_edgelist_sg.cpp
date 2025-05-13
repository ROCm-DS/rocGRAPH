// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "sampling/detail/gather_one_hop_edgelist_impl.cuh"

namespace rocgraph
{
    namespace detail
    {

        template std::tuple<rmm::device_uvector<int32_t>,
                            rmm::device_uvector<int32_t>,
                            std::optional<rmm::device_uvector<float>>,
                            std::optional<rmm::device_uvector<int32_t>>,
                            std::optional<rmm::device_uvector<int32_t>>,
                            std::optional<rmm::device_uvector<int32_t>>>
            gather_one_hop_edgelist(
                raft::handle_t const&                                        handle,
                graph_view_t<int32_t, int32_t, false, false> const&          graph_view,
                std::optional<edge_property_view_t<int32_t, float const*>>   edge_weight_view,
                std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
                std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_edge_type_view,
                raft::device_span<int32_t const>                             active_majors,
                std::optional<raft::device_span<int32_t const>>              active_major_labels,
                bool                                                         do_expensive_check);

        template std::tuple<rmm::device_uvector<int32_t>,
                            rmm::device_uvector<int32_t>,
                            std::optional<rmm::device_uvector<float>>,
                            std::optional<rmm::device_uvector<int64_t>>,
                            std::optional<rmm::device_uvector<int32_t>>,
                            std::optional<rmm::device_uvector<int32_t>>>
            gather_one_hop_edgelist(
                raft::handle_t const&                                        handle,
                graph_view_t<int32_t, int64_t, false, false> const&          graph_view,
                std::optional<edge_property_view_t<int64_t, float const*>>   edge_weight_view,
                std::optional<edge_property_view_t<int64_t, int64_t const*>> edge_id_view,
                std::optional<edge_property_view_t<int64_t, int32_t const*>> edge_edge_type_view,
                raft::device_span<int32_t const>                             active_majors,
                std::optional<raft::device_span<int32_t const>>              active_major_labels,
                bool                                                         do_expensive_check);

        template std::tuple<rmm::device_uvector<int64_t>,
                            rmm::device_uvector<int64_t>,
                            std::optional<rmm::device_uvector<float>>,
                            std::optional<rmm::device_uvector<int64_t>>,
                            std::optional<rmm::device_uvector<int32_t>>,
                            std::optional<rmm::device_uvector<int32_t>>>
            gather_one_hop_edgelist(
                raft::handle_t const&                                        handle,
                graph_view_t<int64_t, int64_t, false, false> const&          graph_view,
                std::optional<edge_property_view_t<int64_t, float const*>>   edge_weight_view,
                std::optional<edge_property_view_t<int64_t, int64_t const*>> edge_id_view,
                std::optional<edge_property_view_t<int64_t, int32_t const*>> edge_edge_type_view,
                raft::device_span<int64_t const>                             active_majors,
                std::optional<raft::device_span<int32_t const>>              active_major_labels,
                bool                                                         do_expensive_check);

        template std::tuple<rmm::device_uvector<int32_t>,
                            rmm::device_uvector<int32_t>,
                            std::optional<rmm::device_uvector<double>>,
                            std::optional<rmm::device_uvector<int32_t>>,
                            std::optional<rmm::device_uvector<int32_t>>,
                            std::optional<rmm::device_uvector<int32_t>>>
            gather_one_hop_edgelist(
                raft::handle_t const&                                        handle,
                graph_view_t<int32_t, int32_t, false, false> const&          graph_view,
                std::optional<edge_property_view_t<int32_t, double const*>>  edge_weight_view,
                std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
                std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_edge_type_view,
                raft::device_span<int32_t const>                             active_majors,
                std::optional<raft::device_span<int32_t const>>              active_major_labels,
                bool                                                         do_expensive_check);

        template std::tuple<rmm::device_uvector<int32_t>,
                            rmm::device_uvector<int32_t>,
                            std::optional<rmm::device_uvector<double>>,
                            std::optional<rmm::device_uvector<int64_t>>,
                            std::optional<rmm::device_uvector<int32_t>>,
                            std::optional<rmm::device_uvector<int32_t>>>
            gather_one_hop_edgelist(
                raft::handle_t const&                                        handle,
                graph_view_t<int32_t, int64_t, false, false> const&          graph_view,
                std::optional<edge_property_view_t<int64_t, double const*>>  edge_weight_view,
                std::optional<edge_property_view_t<int64_t, int64_t const*>> edge_id_view,
                std::optional<edge_property_view_t<int64_t, int32_t const*>> edge_edge_type_view,
                raft::device_span<int32_t const>                             active_majors,
                std::optional<raft::device_span<int32_t const>>              active_major_labels,
                bool                                                         do_expensive_check);

        template std::tuple<rmm::device_uvector<int64_t>,
                            rmm::device_uvector<int64_t>,
                            std::optional<rmm::device_uvector<double>>,
                            std::optional<rmm::device_uvector<int64_t>>,
                            std::optional<rmm::device_uvector<int32_t>>,
                            std::optional<rmm::device_uvector<int32_t>>>
            gather_one_hop_edgelist(
                raft::handle_t const&                                        handle,
                graph_view_t<int64_t, int64_t, false, false> const&          graph_view,
                std::optional<edge_property_view_t<int64_t, double const*>>  edge_weight_view,
                std::optional<edge_property_view_t<int64_t, int64_t const*>> edge_id_view,
                std::optional<edge_property_view_t<int64_t, int32_t const*>> edge_edge_type_view,
                raft::device_span<int64_t const>                             active_majors,
                std::optional<raft::device_span<int32_t const>>              active_major_labels,
                bool                                                         do_expensive_check);

    } // namespace detail
} // namespace rocgraph
