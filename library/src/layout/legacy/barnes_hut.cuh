// Copyright (C) 2020-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "bh_kernels.cuh"
#include "converters/legacy/COOtoCSR.cuh"
#include "fa2_kernels.cuh"
#include "utilities/graph_utils.cuh"
#include "utils.hpp"

#include "detail/utility_wrappers.hpp"
#include "legacy/graph.hpp"
#include "legacy/internals.hpp"
#include "utilities/error.hpp"

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/reduce.h>

#include "control.h"
#include <hip/hip_runtime_api.h>

namespace rocgraph
{
    namespace detail
    {

        template <typename vertex_t, typename edge_t, typename weight_t>
        void barnes_hut(raft::handle_t const&                             handle,
                        legacy::GraphCOOView<vertex_t, edge_t, weight_t>& graph,
                        float*                                            pos,
                        const int                                         max_iter = 500,
                        float*                                            x_start  = nullptr,
                        float*                                            y_start  = nullptr,
                        bool        outbound_attraction_distribution               = true,
                        bool        lin_log_mode                                   = false,
                        bool        prevent_overlapping                            = false,
                        const float edge_weight_influence                          = 1.0,
                        const float jitter_tolerance                               = 1.0,
                        const float theta                                          = 0.5,
                        const float scaling_ratio                                  = 2.0,
                        bool        strong_gravity_mode                            = false,
                        const float gravity                                        = 1.0,
                        bool        verbose                                        = false,
                        internals::GraphBasedDimRedCallback* callback              = nullptr)
        {
            rmm::cuda_stream_view stream_view(handle.get_stream());
            const edge_t          e = graph.number_of_edges;
            const vertex_t        n = graph.number_of_vertices;

            const int blocks = getMultiProcessorCount();
            // A tiny jitter to promote numerical stability/
            const float epssq = 0.0025;
            // We use the same array for nodes and cells.
            int nnodes = n * 2;
            if(nnodes < 1024 * blocks)
                nnodes = 1024 * blocks;
            while((nnodes & (32 - 1)) != 0)
                nnodes++;
            nnodes--;

            // Allocate more space
            //---------------------------------------------------
            rmm::device_uvector<unsigned> d_limiter(1, stream_view);
            rmm::device_uvector<int>      d_maxdepthd(1, stream_view);
            rmm::device_uvector<int>      d_bottomd(1, stream_view);
            rmm::device_uvector<float>    d_radiusd(1, stream_view);

            unsigned* limiter   = d_limiter.data();
            int*      maxdepthd = d_maxdepthd.data();
            int*      bottomd   = d_bottomd.data();
            float*    radiusd   = d_radiusd.data();

            THROW_IF_HIPLAUNCHKERNELGGL_ERROR((InitializationKernel),
                                              dim3(1),
                                              dim3(1),
                                              0,
                                              stream_view.value(),
                                              limiter,
                                              maxdepthd,
                                              radiusd);
            RAFT_CHECK_CUDA(stream_view.value());

            const int   FOUR_NNODES   = 4 * nnodes;
            const int   FOUR_N        = 4 * n;
            const float theta_squared = theta * theta;
            const int   NNODES        = nnodes;

            rmm::device_uvector<int> d_startl(nnodes + 1, stream_view);
            rmm::device_uvector<int> d_childl((nnodes + 1) * 4, stream_view);
            // FA2 requires degree + 1
            rmm::device_uvector<int> d_massl(nnodes + 1, stream_view);
            thrust::fill(handle.get_thrust_policy(), d_massl.begin(), d_massl.end(), 1);

            rmm::device_uvector<float> d_maxxl(blocks * FACTOR1, stream_view);
            rmm::device_uvector<float> d_maxyl(blocks * FACTOR1, stream_view);
            rmm::device_uvector<float> d_minxl(blocks * FACTOR1, stream_view);
            rmm::device_uvector<float> d_minyl(blocks * FACTOR1, stream_view);

            // Actual mallocs
            int* startl = d_startl.data();
            int* childl = d_childl.data();
            int* massl  = d_massl.data();

            float* maxxl = d_maxxl.data();
            float* maxyl = d_maxyl.data();
            float* minxl = d_minxl.data();
            float* minyl = d_minyl.data();

            // SummarizationKernel
            rmm::device_uvector<int> d_countl(nnodes + 1, stream_view);
            int*                     countl = d_countl.data();

            // SortKernel
            rmm::device_uvector<int> d_sortl(nnodes + 1, stream_view);
            int*                     sortl = d_sortl.data();

            // RepulsionKernel
            rmm::device_uvector<float> d_rep_forces((nnodes + 1) * 2, stream_view);
            float*                     rep_forces = d_rep_forces.data();

            rmm::device_uvector<float> d_radius_squared(1, stream_view);
            float*                     radiusd_squared = d_radius_squared.data();

            rmm::device_uvector<float> d_nodes_pos((nnodes + 1) * 2, stream_view);
            float*                     nodes_pos = d_nodes_pos.data();

            // Initialize positions with random values
            raft::random::RngState rng_state{0};

            // Copy start x and y positions.
            if(x_start && y_start)
            {
                raft::copy(nodes_pos, x_start, n, stream_view.value());
                raft::copy(nodes_pos + nnodes + 1, y_start, n, stream_view.value());
            }
            else
            {
                uniform_random_fill(
                    handle.get_stream(), nodes_pos, (nnodes + 1) * 2, -100.0f, 100.0f, rng_state);
            }

            // Allocate arrays for force computation
            float* attract{nullptr};
            float* old_forces{nullptr};
            float* swinging{nullptr};
            float* traction{nullptr};

            rmm::device_uvector<float> d_attract(n * 2, stream_view);
            rmm::device_uvector<float> d_old_forces(n * 2, stream_view);
            rmm::device_uvector<float> d_swinging(n, stream_view);
            rmm::device_uvector<float> d_traction(n, stream_view);

            attract    = d_attract.data();
            old_forces = d_old_forces.data();
            swinging   = d_swinging.data();
            traction   = d_traction.data();

            thrust::fill(handle.get_thrust_policy(), d_old_forces.begin(), d_old_forces.end(), 0.f);

            // Sort COO for coalesced memory access.
            sort(graph, stream_view.value());
            RAFT_CHECK_CUDA(stream_view.value());

            graph.degree(massl, rocgraph::legacy::DegreeDirection::OUT);
            RAFT_CHECK_CUDA(stream_view.value());

            const vertex_t* row = graph.src_indices;
            const vertex_t* col = graph.dst_indices;
            const weight_t* v   = graph.edge_data;

            // Scalars used to adapt global speed.
            float speed                     = 1.f;
            float speed_efficiency          = 1.f;
            float outbound_att_compensation = 1.f;
            float jt                        = 0.f;

            // If outboundAttractionDistribution active, compensate.
            if(outbound_attraction_distribution)
            {
                int sum = thrust::reduce(
                    handle.get_thrust_policy(), d_massl.begin(), d_massl.begin() + n);
                outbound_att_compensation = sum / (float)n;
            }

            //
            // Set cache levels for faster algorithm execution
            //---------------------------------------------------
            hipFuncSetCacheConfig(reinterpret_cast<const void*>(&BoundingBoxKernel),
                                  hipFuncCachePreferShared);
            hipFuncSetCacheConfig(reinterpret_cast<const void*>(&TreeBuildingKernel),
                                  hipFuncCachePreferL1);
            hipFuncSetCacheConfig(reinterpret_cast<const void*>(&ClearKernel1),
                                  hipFuncCachePreferL1);
            hipFuncSetCacheConfig(reinterpret_cast<const void*>(&ClearKernel2),
                                  hipFuncCachePreferL1);
            hipFuncSetCacheConfig(reinterpret_cast<const void*>(&SummarizationKernel),
                                  hipFuncCachePreferShared);
            hipFuncSetCacheConfig(reinterpret_cast<const void*>(&SortKernel), hipFuncCachePreferL1);
            hipFuncSetCacheConfig(reinterpret_cast<const void*>(&RepulsionKernel),
                                  hipFuncCachePreferL1);
            hipFuncSetCacheConfig(reinterpret_cast<const void*>(&apply_forces_bh),
                                  hipFuncCachePreferL1);

            if(callback)
            {
                callback->setup<float>(nnodes + 1, 2);
                callback->on_preprocess_end(nodes_pos);
            }

            for(int iter = 0; iter < max_iter; ++iter)
            {
                // Reset force values
                thrust::fill(
                    handle.get_thrust_policy(), d_rep_forces.begin(), d_rep_forces.end(), 0.f);
                thrust::fill(handle.get_thrust_policy(), d_attract.begin(), d_attract.end(), 0.f);
                thrust::fill(handle.get_thrust_policy(), d_swinging.begin(), d_swinging.end(), 0.f);
                thrust::fill(handle.get_thrust_policy(), d_traction.begin(), d_traction.end(), 0.f);

                THROW_IF_HIPLAUNCHKERNELGGL_ERROR((ResetKernel),
                                                  dim3(1),
                                                  dim3(1),
                                                  0,
                                                  stream_view.value(),
                                                  radiusd_squared,
                                                  bottomd,
                                                  NNODES,
                                                  radiusd);
                RAFT_CHECK_CUDA(stream_view.value());

                // Compute bounding box arround all bodies
                THROW_IF_HIPLAUNCHKERNELGGL_ERROR((BoundingBoxKernel),
                                                  dim3(blocks * FACTOR1),
                                                  dim3(THREADS1),
                                                  0,
                                                  stream_view.value(),
                                                  startl,
                                                  childl,
                                                  massl,
                                                  nodes_pos,
                                                  nodes_pos + nnodes + 1,
                                                  maxxl,
                                                  maxyl,
                                                  minxl,
                                                  minyl,
                                                  FOUR_NNODES,
                                                  NNODES,
                                                  n,
                                                  limiter,
                                                  radiusd);
                RAFT_CHECK_CUDA(stream_view.value());

                THROW_IF_HIPLAUNCHKERNELGGL_ERROR((ClearKernel1),
                                                  dim3(blocks),
                                                  dim3(1024),
                                                  0,
                                                  stream_view.value(),
                                                  childl,
                                                  FOUR_NNODES,
                                                  FOUR_N);
                RAFT_CHECK_CUDA(stream_view.value());

                // Build quadtree
                THROW_IF_HIPLAUNCHKERNELGGL_ERROR((TreeBuildingKernel),
                                                  dim3(blocks * FACTOR2),
                                                  dim3(THREADS2),
                                                  0,
                                                  stream_view.value(),
                                                  childl,
                                                  nodes_pos,
                                                  nodes_pos + nnodes + 1,
                                                  NNODES,
                                                  n,
                                                  maxdepthd,
                                                  bottomd,
                                                  radiusd);
                RAFT_CHECK_CUDA(stream_view.value());

                THROW_IF_HIPLAUNCHKERNELGGL_ERROR((ClearKernel2),
                                                  dim3(blocks),
                                                  dim3(1024),
                                                  0,
                                                  stream_view.value(),
                                                  startl,
                                                  massl,
                                                  NNODES,
                                                  bottomd);
                RAFT_CHECK_CUDA(stream_view.value());

                // Summarizes mass and position for each cell, bottom up approach
                THROW_IF_HIPLAUNCHKERNELGGL_ERROR((SummarizationKernel),
                                                  dim3(blocks * FACTOR3),
                                                  dim3(THREADS3),
                                                  0,
                                                  stream_view.value(),
                                                  countl,
                                                  childl,
                                                  massl,
                                                  nodes_pos,
                                                  nodes_pos + nnodes + 1,
                                                  NNODES,
                                                  n,
                                                  bottomd);
                RAFT_CHECK_CUDA(stream_view.value());

                // Group closed bodies together, used to speed up Repulsion kernel
                THROW_IF_HIPLAUNCHKERNELGGL_ERROR((SortKernel),
                                                  dim3(blocks * FACTOR4),
                                                  dim3(THREADS4),
                                                  0,
                                                  stream_view.value(),
                                                  sortl,
                                                  countl,
                                                  startl,
                                                  childl,
                                                  NNODES,
                                                  n,
                                                  bottomd);
                RAFT_CHECK_CUDA(stream_view.value());

                // Force computation O(n . log(n))
                THROW_IF_HIPLAUNCHKERNELGGL_ERROR((RepulsionKernel),
                                                  dim3(blocks * FACTOR5),
                                                  dim3(THREADS5),
                                                  0,
                                                  stream_view.value(),
                                                  scaling_ratio,
                                                  theta,
                                                  epssq,
                                                  sortl,
                                                  childl,
                                                  massl,
                                                  nodes_pos,
                                                  nodes_pos + nnodes + 1,
                                                  rep_forces,
                                                  rep_forces + nnodes + 1,
                                                  theta_squared,
                                                  NNODES,
                                                  FOUR_NNODES,
                                                  n,
                                                  radiusd_squared,
                                                  maxdepthd);
                RAFT_CHECK_CUDA(stream_view.value());

                apply_gravity<vertex_t>(nodes_pos,
                                        nodes_pos + nnodes + 1,
                                        attract,
                                        attract + n,
                                        massl,
                                        gravity,
                                        strong_gravity_mode,
                                        scaling_ratio,
                                        n,
                                        stream_view.value());

                apply_attraction<vertex_t, edge_t, weight_t>(row,
                                                             col,
                                                             v,
                                                             e,
                                                             nodes_pos,
                                                             nodes_pos + nnodes + 1,
                                                             attract,
                                                             attract + n,
                                                             massl,
                                                             outbound_attraction_distribution,
                                                             lin_log_mode,
                                                             edge_weight_influence,
                                                             outbound_att_compensation,
                                                             stream_view.value());

                compute_local_speed(rep_forces,
                                    rep_forces + nnodes + 1,
                                    attract,
                                    attract + n,
                                    old_forces,
                                    old_forces + n,
                                    massl,
                                    swinging,
                                    traction,
                                    n,
                                    stream_view.value());

                // Compute global swinging and traction values
                const float s = thrust::reduce(
                    handle.get_thrust_policy(), d_swinging.begin(), d_swinging.end());

                const float t = thrust::reduce(
                    handle.get_thrust_policy(), d_traction.begin(), d_traction.end());

                // Compute global speed based on gloab and local swinging and traction.
                adapt_speed<vertex_t>(jitter_tolerance, &jt, &speed, &speed_efficiency, s, t, n);

                // Update positions
                THROW_IF_HIPLAUNCHKERNELGGL_ERROR((apply_forces_bh),
                                                  dim3(blocks * FACTOR6),
                                                  dim3(THREADS6),
                                                  0,
                                                  stream_view.value(),
                                                  nodes_pos,
                                                  nodes_pos + nnodes + 1,
                                                  attract,
                                                  attract + n,
                                                  rep_forces,
                                                  rep_forces + nnodes + 1,
                                                  old_forces,
                                                  old_forces + n,
                                                  swinging,
                                                  speed,
                                                  n);

                if(callback)
                    callback->on_epoch_end(nodes_pos);

                if(verbose)
                {
                    std::cout << "iteration: " << iter + 1 << ", speed: " << speed
                              << ", speed_efficiency: " << speed_efficiency << ", jt: " << jt
                              << ", swinging: " << s << ", traction: " << t << "\n";
                }
            }

            // Copy nodes positions into final output pos
            raft::copy(pos, nodes_pos, n, stream_view.value());
            raft::copy(pos + n, nodes_pos + nnodes + 1, n, stream_view.value());

            if(callback)
                callback->on_train_end(nodes_pos);
        }

    } // namespace detail
} // namespace rocgraph
