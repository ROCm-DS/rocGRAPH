// Copyright (C) 2020-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "barnes_hut.cuh"
#include "exact_fa2.cuh"

namespace rocgraph
{

    template <typename vertex_t, typename edge_t, typename weight_t>
    void force_atlas2(raft::handle_t const&                             handle,
                      legacy::GraphCOOView<vertex_t, edge_t, weight_t>& graph,
                      float*                                            pos,
                      const int                                         max_iter,
                      float*                                            x_start,
                      float*                                            y_start,
                      bool                                 outbound_attraction_distribution,
                      bool                                 lin_log_mode,
                      bool                                 prevent_overlapping,
                      const float                          edge_weight_influence,
                      const float                          jitter_tolerance,
                      bool                                 barnes_hut_optimize,
                      const float                          barnes_hut_theta,
                      const float                          scaling_ratio,
                      bool                                 strong_gravity_mode,
                      const float                          gravity,
                      bool                                 verbose,
                      internals::GraphBasedDimRedCallback* callback)
    {
        ROCGRAPH_EXPECTS(pos != nullptr,
                         "Invalid input argument: pos array should be of size 2 * V");
        ROCGRAPH_EXPECTS(graph.number_of_vertices != 0, "Invalid input: Graph is empty");

        if(!barnes_hut_optimize)
        {
            rocgraph::detail::exact_fa2<vertex_t, edge_t, weight_t>(
                handle,
                graph,
                pos,
                max_iter,
                x_start,
                y_start,
                outbound_attraction_distribution,
                lin_log_mode,
                prevent_overlapping,
                edge_weight_influence,
                jitter_tolerance,
                scaling_ratio,
                strong_gravity_mode,
                gravity,
                verbose,
                callback);
        }
        else
        {
            rocgraph::detail::barnes_hut<vertex_t, edge_t, weight_t>(
                handle,
                graph,
                pos,
                max_iter,
                x_start,
                y_start,
                outbound_attraction_distribution,
                lin_log_mode,
                prevent_overlapping,
                edge_weight_influence,
                jitter_tolerance,
                barnes_hut_theta,
                scaling_ratio,
                strong_gravity_mode,
                gravity,
                verbose,
                callback);
        }
    }

    template void force_atlas2<int, int, float>(raft::handle_t const&                  handle,
                                                legacy::GraphCOOView<int, int, float>& graph,
                                                float*                                 pos,
                                                const int                              max_iter,
                                                float*                                 x_start,
                                                float*                                 y_start,
                                                bool        outbound_attraction_distribution,
                                                bool        lin_log_mode,
                                                bool        prevent_overlapping,
                                                const float edge_weight_influence,
                                                const float jitter_tolerance,
                                                bool        barnes_hut_optimize,
                                                const float barnes_hut_theta,
                                                const float scaling_ratio,
                                                bool        strong_gravity_mode,
                                                const float gravity,
                                                bool        verbose,
                                                internals::GraphBasedDimRedCallback* callback);

    template void force_atlas2<int, int, double>(raft::handle_t const&                   handle,
                                                 legacy::GraphCOOView<int, int, double>& graph,
                                                 float*                                  pos,
                                                 const int                               max_iter,
                                                 float*                                  x_start,
                                                 float*                                  y_start,
                                                 bool        outbound_attraction_distribution,
                                                 bool        lin_log_mode,
                                                 bool        prevent_overlapping,
                                                 const float edge_weight_influence,
                                                 const float jitter_tolerance,
                                                 bool        barnes_hut_optimize,
                                                 const float barnes_hut_theta,
                                                 const float scaling_ratio,
                                                 bool        strong_gravity_mode,
                                                 const float gravity,
                                                 bool        verbose,
                                                 internals::GraphBasedDimRedCallback* callback);

} // namespace rocgraph
