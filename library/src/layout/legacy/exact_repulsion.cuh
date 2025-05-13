#include "hip/hip_runtime.h"

// Copyright (C) 2020-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#define restrict __restrict__
#define CUDA_MAX_BLOCKS_2D 256

namespace rocgraph
{
    namespace detail
    {

        template <typename vertex_t>
        __global__ static void repulsion_kernel(const float* restrict x_pos,
                                                const float* restrict y_pos,
                                                float* restrict repel_x,
                                                float* restrict repel_y,
                                                const int* restrict mass,
                                                const float    scaling_ratio,
                                                const vertex_t n)
        {
            int j = (blockIdx.x * blockDim.x) + threadIdx.x; // for every item in row
            int i = (blockIdx.y * blockDim.y) + threadIdx.y; // for every row
            for(; i < n; i += gridDim.y * blockDim.y)
            {
                for(; j < n; j += gridDim.x * blockDim.x)
                {
                    if(j >= i)
                        return;
                    float x_dist   = x_pos[i] - x_pos[j];
                    float y_dist   = y_pos[i] - y_pos[j];
                    float distance = x_dist * x_dist + y_dist * y_dist;
                    distance += FLT_EPSILON;
                    float factor = scaling_ratio * mass[i] * mass[j] / distance;
                    // Add forces
                    atomicAdd(&repel_x[i], x_dist * factor);
                    atomicAdd(&repel_x[j], -x_dist * factor);
                    atomicAdd(&repel_y[i], y_dist * factor);
                    atomicAdd(&repel_y[j], -y_dist * factor);
                }
            }
        }

        template <typename vertex_t, int TPB_X = 32, int TPB_Y = 32>
        void apply_repulsion(const float* restrict x_pos,
                             const float* restrict y_pos,
                             float* restrict repel_x,
                             float* restrict repel_y,
                             const int* restrict mass,
                             const float    scaling_ratio,
                             const vertex_t n,
                             hipStream_t    stream)
        {
            dim3 nthreads(TPB_X, TPB_Y);
            dim3 nblocks(min((n + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS_2D),
                         min((n + nthreads.y - 1) / nthreads.y, CUDA_MAX_BLOCKS_2D));

            THROW_IF_HIPLAUNCHKERNELGGL_ERROR((repulsion_kernel<vertex_t>),
                                              nblocks,
                                              nthreads,
                                              0,
                                              stream,
                                              x_pos,
                                              y_pos,
                                              repel_x,
                                              repel_y,
                                              mass,
                                              scaling_ratio,
                                              n);
            RAFT_CHECK_CUDA(stream);
        }

    } // namespace detail
} // namespace rocgraph
