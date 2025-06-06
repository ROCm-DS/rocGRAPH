// Copyright (c) 2023, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <raft/util/cuda_dev_essentials.cuh> // DI

namespace raft::distance::detail::ops
{

    // Describes the computation the template distance
    //
    // Fill in the TODO items.

    template <typename DataType, typename AccType, typename IdxType>
    struct template_distance_op
    {
        using DataT = DataType;
        using AccT  = AccType;
        using IdxT  = IdxType;

        TODO member;

        template_distance_op(TODO member_) noexcept
            : member(member_)
        {
        }

        // Load norms of input data
        static constexpr bool use_norms = TODO;
        // Whether the core function requires so many instructions that it makes sense
        // to reduce loop unrolling, etc. We do this to keep compile times in check.
        static constexpr bool expensive_inner_loop = false;

        // Size of shared memory. This is normally decided by the kernel policy, but
        // some ops such as correlation_distance_op use more.
        template <typename Policy>
        static constexpr size_t shared_mem_size()
        {
            return Policy::SmemSize + TODO;
        }

        DI void core(AccT& acc, DataT& x, DataT& y) const
        {
            TODO;
        };

        template <typename Policy>
        DI void epilog(AccT   acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                       DataT* regxn,
                       DataT* regyn,
                       IdxT   gridStrideX,
                       IdxT   gridStrideY) const
        {
            TODO;
        }

        // If exist, returns a cutlass op that performs the same operation.
        // See cosine and l2_exp distance ops for an example.
        constexpr l2_exp_cutlass_op<DataT, AccT> get_cutlass_op() const
        {
            TODO;
        }
    };

} // namespace raft::distance::detail::ops
