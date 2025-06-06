// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "cores/k_core_impl.cuh"

namespace rocgraph
{

    // SG instantiation

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<float>>>
        k_core(raft::handle_t const&                                      handle,
               graph_view_t<int32_t, int32_t, false, false> const&        graph_view,
               std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
               size_t                                                     k,
               std::optional<k_core_degree_type_t>                        degree_type,
               std::optional<raft::device_span<int32_t const>>            core_numbers,
               bool                                                       do_expensive_check);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<float>>>
        k_core(raft::handle_t const&                                      handle,
               graph_view_t<int32_t, int64_t, false, false> const&        graph_view,
               std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
               size_t                                                     k,
               std::optional<k_core_degree_type_t>                        degree_type,
               std::optional<raft::device_span<int64_t const>>            core_numbers,
               bool                                                       do_expensive_check);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>,
                        std::optional<rmm::device_uvector<float>>>
        k_core(raft::handle_t const&                                      handle,
               graph_view_t<int64_t, int64_t, false, false> const&        graph_view,
               std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
               size_t                                                     k,
               std::optional<k_core_degree_type_t>                        degree_type,
               std::optional<raft::device_span<int64_t const>>            core_numbers,
               bool                                                       do_expensive_check);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<double>>>
        k_core(raft::handle_t const&                                       handle,
               graph_view_t<int32_t, int32_t, false, false> const&         graph_view,
               std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
               size_t                                                      k,
               std::optional<k_core_degree_type_t>                         degree_type,
               std::optional<raft::device_span<int32_t const>>             core_numbers,
               bool                                                        do_expensive_check);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<double>>>
        k_core(raft::handle_t const&                                       handle,
               graph_view_t<int32_t, int64_t, false, false> const&         graph_view,
               std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
               size_t                                                      k,
               std::optional<k_core_degree_type_t>                         degree_type,
               std::optional<raft::device_span<int64_t const>>             core_numbers,
               bool                                                        do_expensive_check);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>,
                        std::optional<rmm::device_uvector<double>>>
        k_core(raft::handle_t const&                                       handle,
               graph_view_t<int64_t, int64_t, false, false> const&         graph_view,
               std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
               size_t                                                      k,
               std::optional<k_core_degree_type_t>                         degree_type,
               std::optional<raft::device_span<int64_t const>>             core_numbers,
               bool                                                        do_expensive_check);

} // namespace rocgraph
