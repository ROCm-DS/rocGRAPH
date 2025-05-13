// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "link_analysis/hits_impl.cuh"

namespace rocgraph
{

    // MG instantiation
    template std::tuple<float, size_t>
        hits(raft::handle_t const&                             handle,
             graph_view_t<int32_t, int32_t, true, true> const& graph_view,
             float* const                                      hubs,
             float* const                                      authorities,
             float                                             epsilon,
             size_t                                            max_iterations,
             bool                                              has_initial_hubs_guess,
             bool                                              normalize,
             bool                                              do_expensive_check);

    template std::tuple<double, size_t>
        hits(raft::handle_t const&                             handle,
             graph_view_t<int32_t, int32_t, true, true> const& graph_view,
             double* const                                     hubs,
             double* const                                     authorities,
             double                                            epsilon,
             size_t                                            max_iterations,
             bool                                              has_initial_hubs_guess,
             bool                                              normalize,
             bool                                              do_expensive_check);

    template std::tuple<float, size_t>
        hits(raft::handle_t const&                             handle,
             graph_view_t<int32_t, int64_t, true, true> const& graph_view,
             float* const                                      hubs,
             float* const                                      authorities,
             float                                             epsilon,
             size_t                                            max_iterations,
             bool                                              has_initial_hubs_guess,
             bool                                              normalize,
             bool                                              do_expensive_check);

    template std::tuple<double, size_t>
        hits(raft::handle_t const&                             handle,
             graph_view_t<int32_t, int64_t, true, true> const& graph_view,
             double* const                                     hubs,
             double* const                                     authorities,
             double                                            epsilon,
             size_t                                            max_iterations,
             bool                                              has_initial_hubs_guess,
             bool                                              normalize,
             bool                                              do_expensive_check);

    template std::tuple<float, size_t>
        hits(raft::handle_t const&                             handle,
             graph_view_t<int64_t, int64_t, true, true> const& graph_view,
             float* const                                      hubs,
             float* const                                      authorities,
             float                                             epsilon,
             size_t                                            max_iterations,
             bool                                              has_initial_hubs_guess,
             bool                                              normalize,
             bool                                              do_expensive_check);

    template std::tuple<double, size_t>
        hits(raft::handle_t const&                             handle,
             graph_view_t<int64_t, int64_t, true, true> const& graph_view,
             double* const                                     hubs,
             double* const                                     authorities,
             double                                            epsilon,
             size_t                                            max_iterations,
             bool                                              has_initial_hubs_guess,
             bool                                              normalize,
             bool                                              do_expensive_check);

} // namespace rocgraph
