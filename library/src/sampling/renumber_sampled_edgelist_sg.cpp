// Copyright (C) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "renumber_sampled_edgelist_impl.cuh"

#include "sampling_functions.hpp"

// FIXME: deprecated, to be deleted
namespace rocgraph
{

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        std::optional<rmm::device_uvector<size_t>>>
        renumber_sampled_edgelist(
            raft::handle_t const&                                      handle,
            rmm::device_uvector<int32_t>&&                             edgelist_srcs,
            rmm::device_uvector<int32_t>&&                             edgelist_dsts,
            std::optional<raft::device_span<int32_t const>>            edgelist_hops,
            std::optional<std::tuple<raft::device_span<int32_t const>,
                                     raft::device_span<size_t const>>> label_offsets,
            bool                                                       do_expensive_check);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>,
                        std::optional<rmm::device_uvector<size_t>>>
        renumber_sampled_edgelist(
            raft::handle_t const&                                      handle,
            rmm::device_uvector<int64_t>&&                             edgelist_srcs,
            rmm::device_uvector<int64_t>&&                             edgelist_dsts,
            std::optional<raft::device_span<int32_t const>>            edgelist_hops,
            std::optional<std::tuple<raft::device_span<int32_t const>,
                                     raft::device_span<size_t const>>> label_offsets,
            bool                                                       do_expensive_check);

} // namespace rocgraph
