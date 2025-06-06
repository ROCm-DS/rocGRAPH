// Copyright (c) 2023, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <raft/util/cuda_dev_essentials.cuh> // DI

namespace raft::distance::detail::ops
{

    /**
 * @brief the L_inf (Chebyshev) distance matrix calculation
 *
 * It computes the following equation:
 *
 *  c_ij = max_k | x_ik - y_kj |
 */
    template <typename DataType, typename AccType, typename IdxType>
    struct l_inf_distance_op
    {
        using DataT = DataType;
        using AccT  = AccType;
        using IdxT  = IdxType;

        // Load norms of input data
        static constexpr bool use_norms = false;
        // Whether the core function requires so many instructions that it makes sense
        // to reduce loop unrolling, etc. We do this to keep compile times in check.
        static constexpr bool expensive_inner_loop = false;

        // Size of shared memory. This is normally decided by the kernel policy, but
        // some ops such as correlation_distance_op use more.
        template <typename Policy>
        static constexpr size_t shared_mem_size()
        {
            return Policy::SmemSize;
        }

        DI void core(AccT& acc, DataT& x, DataT& y) const
        {
            const auto diff = raft::abs(x - y);
            acc             = raft::max(acc, diff);
        };

        template <typename Policy>
        DI void epilog(AccT   acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                       DataT* regxn,
                       DataT* regyn,
                       IdxT   gridStrideX,
                       IdxT   gridStrideY) const
        {
            return;
        }
    };

} // namespace raft::distance::detail::ops
