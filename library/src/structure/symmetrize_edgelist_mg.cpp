// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "structure/symmetrize_edgelist_impl.cuh"

namespace rocgraph
{

    // MG instantiation

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<float>>>
        symmetrize_edgelist<int32_t, float, false, true>(
            raft::handle_t const&                       handle,
            rmm::device_uvector<int32_t>&&              edgelist_srcs,
            rmm::device_uvector<int32_t>&&              edgelist_dsts,
            std::optional<rmm::device_uvector<float>>&& edgelist_weights,
            bool                                        reciprocal);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<float>>>
        symmetrize_edgelist<int32_t, float, true, true>(
            raft::handle_t const&                       handle,
            rmm::device_uvector<int32_t>&&              edgelist_srcs,
            rmm::device_uvector<int32_t>&&              edgelist_dsts,
            std::optional<rmm::device_uvector<float>>&& edgelist_weights,
            bool                                        reciprocal);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<double>>>
        symmetrize_edgelist<int32_t, double, false, true>(
            raft::handle_t const&                        handle,
            rmm::device_uvector<int32_t>&&               edgelist_srcs,
            rmm::device_uvector<int32_t>&&               edgelist_dsts,
            std::optional<rmm::device_uvector<double>>&& edgelist_weights,
            bool                                         reciprocal);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<double>>>
        symmetrize_edgelist<int32_t, double, true, true>(
            raft::handle_t const&                        handle,
            rmm::device_uvector<int32_t>&&               edgelist_srcs,
            rmm::device_uvector<int32_t>&&               edgelist_dsts,
            std::optional<rmm::device_uvector<double>>&& edgelist_weights,
            bool                                         reciprocal);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>,
                        std::optional<rmm::device_uvector<float>>>
        symmetrize_edgelist<int64_t, float, false, true>(
            raft::handle_t const&                       handle,
            rmm::device_uvector<int64_t>&&              edgelist_srcs,
            rmm::device_uvector<int64_t>&&              edgelist_dsts,
            std::optional<rmm::device_uvector<float>>&& edgelist_weights,
            bool                                        reciprocal);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>,
                        std::optional<rmm::device_uvector<float>>>
        symmetrize_edgelist<int64_t, float, true, true>(
            raft::handle_t const&                       handle,
            rmm::device_uvector<int64_t>&&              edgelist_srcs,
            rmm::device_uvector<int64_t>&&              edgelist_dsts,
            std::optional<rmm::device_uvector<float>>&& edgelist_weights,
            bool                                        reciprocal);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>,
                        std::optional<rmm::device_uvector<double>>>
        symmetrize_edgelist<int64_t, double, false, true>(
            raft::handle_t const&                        handle,
            rmm::device_uvector<int64_t>&&               edgelist_srcs,
            rmm::device_uvector<int64_t>&&               edgelist_dsts,
            std::optional<rmm::device_uvector<double>>&& edgelist_weights,
            bool                                         reciprocal);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>,
                        std::optional<rmm::device_uvector<double>>>
        symmetrize_edgelist<int64_t, double, true, true>(
            raft::handle_t const&                        handle,
            rmm::device_uvector<int64_t>&&               edgelist_srcs,
            rmm::device_uvector<int64_t>&&               edgelist_dsts,
            std::optional<rmm::device_uvector<double>>&& edgelist_weights,
            bool                                         reciprocal);

} // namespace rocgraph
