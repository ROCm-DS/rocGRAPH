// Copyright (C) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "structure/remove_self_loops_impl.cuh"

namespace rocgraph
{

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<float>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<int32_t>>>
        remove_self_loops(raft::handle_t const&                         handle,
                          rmm::device_uvector<int32_t>&&                edgelist_srcs,
                          rmm::device_uvector<int32_t>&&                edgelist_dsts,
                          std::optional<rmm::device_uvector<float>>&&   edgelist_weights,
                          std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_ids,
                          std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<float>>,
                        std::optional<rmm::device_uvector<int64_t>>,
                        std::optional<rmm::device_uvector<int32_t>>>
        remove_self_loops(raft::handle_t const&                         handle,
                          rmm::device_uvector<int32_t>&&                edgelist_srcs,
                          rmm::device_uvector<int32_t>&&                edgelist_dsts,
                          std::optional<rmm::device_uvector<float>>&&   edgelist_weights,
                          std::optional<rmm::device_uvector<int64_t>>&& edgelist_edge_ids,
                          std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>,
                        std::optional<rmm::device_uvector<float>>,
                        std::optional<rmm::device_uvector<int64_t>>,
                        std::optional<rmm::device_uvector<int32_t>>>
        remove_self_loops(raft::handle_t const&                         handle,
                          rmm::device_uvector<int64_t>&&                edgelist_srcs,
                          rmm::device_uvector<int64_t>&&                edgelist_dsts,
                          std::optional<rmm::device_uvector<float>>&&   edgelist_weights,
                          std::optional<rmm::device_uvector<int64_t>>&& edgelist_edge_ids,
                          std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<double>>,
                        std::optional<rmm::device_uvector<int32_t>>,
                        std::optional<rmm::device_uvector<int32_t>>>
        remove_self_loops(raft::handle_t const&                         handle,
                          rmm::device_uvector<int32_t>&&                edgelist_srcs,
                          rmm::device_uvector<int32_t>&&                edgelist_dsts,
                          std::optional<rmm::device_uvector<double>>&&  edgelist_weights,
                          std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_ids,
                          std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<double>>,
                        std::optional<rmm::device_uvector<int64_t>>,
                        std::optional<rmm::device_uvector<int32_t>>>
        remove_self_loops(raft::handle_t const&                         handle,
                          rmm::device_uvector<int32_t>&&                edgelist_srcs,
                          rmm::device_uvector<int32_t>&&                edgelist_dsts,
                          std::optional<rmm::device_uvector<double>>&&  edgelist_weights,
                          std::optional<rmm::device_uvector<int64_t>>&& edgelist_edge_ids,
                          std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>,
                        std::optional<rmm::device_uvector<double>>,
                        std::optional<rmm::device_uvector<int64_t>>,
                        std::optional<rmm::device_uvector<int32_t>>>
        remove_self_loops(raft::handle_t const&                         handle,
                          rmm::device_uvector<int64_t>&&                edgelist_srcs,
                          rmm::device_uvector<int64_t>&&                edgelist_dsts,
                          std::optional<rmm::device_uvector<double>>&&  edgelist_weights,
                          std::optional<rmm::device_uvector<int64_t>>&& edgelist_edge_ids,
                          std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types);

} // namespace rocgraph
