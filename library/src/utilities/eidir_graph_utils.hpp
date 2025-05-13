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

        template ROCGRAPH_EXPORT __device__ float
            parallel_prefix_sum(int32_t, int32_t const*, float const*);
        template ROCGRAPH_EXPORT __device__ double
            parallel_prefix_sum(int32_t, int32_t const*, double const*);
        template ROCGRAPH_EXPORT __device__ float
            parallel_prefix_sum(int64_t, int32_t const*, float const*);
        template ROCGRAPH_EXPORT __device__ double
            parallel_prefix_sum(int64_t, int32_t const*, double const*);
        template ROCGRAPH_EXPORT __device__ float
            parallel_prefix_sum(int64_t, int64_t const*, float const*);
        template ROCGRAPH_EXPORT __device__ double
            parallel_prefix_sum(int64_t, int64_t const*, double const*);

        template ROCGRAPH_EXPORT void
            offsets_to_indices<int32_t, int32_t>(int32_t const*, int32_t, int32_t*);
        template ROCGRAPH_EXPORT void
            offsets_to_indices<int64_t, int32_t>(int64_t const*, int32_t, int32_t*);
        template ROCGRAPH_EXPORT void
            offsets_to_indices<int64_t, int64_t>(int64_t const*, int64_t, int64_t*);

        template __global__ __attribute__((visibility("hidden"))) void
            offsets_to_indices_kernel<int32_t, int32_t>(int32_t const*, int32_t, int32_t*);
        template __global__ __attribute__((visibility("hidden"))) void
            offsets_to_indices_kernel<int64_t, int32_t>(int64_t const*, int32_t, int32_t*);
        template __global__ __attribute__((visibility("hidden"))) void
            offsets_to_indices_kernel<int64_t, int64_t>(int64_t const*, int64_t, int64_t*);

    } // namespace detail
} // namespace rocgraph
