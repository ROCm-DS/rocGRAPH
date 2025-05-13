#include "hip/hip_runtime.h"

// Copyright (C) 2020-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph-export.h"

namespace rocgraph
{
    namespace detail
    {

        extern template ROCGRAPH_EXPORT __device__ float
            parallel_prefix_sum(int32_t, int32_t const*, float const*);
        extern template ROCGRAPH_EXPORT __device__ double
            parallel_prefix_sum(int32_t, int32_t const*, double const*);
        extern template ROCGRAPH_EXPORT __device__ float
            parallel_prefix_sum(int64_t, int32_t const*, float const*);
        extern template ROCGRAPH_EXPORT __device__ double
            parallel_prefix_sum(int64_t, int32_t const*, double const*);
        extern template ROCGRAPH_EXPORT __device__ float
            parallel_prefix_sum(int64_t, int64_t const*, float const*);
        extern template ROCGRAPH_EXPORT __device__ double
            parallel_prefix_sum(int64_t, int64_t const*, double const*);

        extern template ROCGRAPH_EXPORT void offsets_to_indices<int, int>(int const*, int, int*);
        extern template ROCGRAPH_EXPORT void offsets_to_indices<long, int>(long const*, int, int*);
        extern template ROCGRAPH_EXPORT void
            offsets_to_indices<long, long>(long const*, long, long*);

        extern template __global__ void offsets_to_indices_kernel<int, int>(int const*, int, int*);
        extern template __global__ void
            offsets_to_indices_kernel<long, int>(long const*, int, int*);
        extern template __global__ void
            offsets_to_indices_kernel<long, long>(long const*, long, long*);

    } // namespace detail
} // namespace rocgraph
