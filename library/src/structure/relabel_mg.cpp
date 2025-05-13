// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "structure/relabel_impl.cuh"

namespace rocgraph
{

    // MG instantiation

    template void
        relabel<int32_t, true>(raft::handle_t const&                      handle,
                               std::tuple<int32_t const*, int32_t const*> old_new_label_pairs,
                               int32_t                                    num_label_pairs,
                               int32_t*                                   labels,
                               int32_t                                    num_labels,
                               bool                                       skip_missing_labels,
                               bool                                       do_expensive_check);

    template void
        relabel<int64_t, true>(raft::handle_t const&                      handle,
                               std::tuple<int64_t const*, int64_t const*> old_new_label_pairs,
                               int64_t                                    num_label_pairs,
                               int64_t*                                   labels,
                               int64_t                                    num_labels,
                               bool                                       skip_missing_labels,
                               bool                                       do_expensive_check);

} // namespace rocgraph
