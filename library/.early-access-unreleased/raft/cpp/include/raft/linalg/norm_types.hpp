// Copyright (c) 2022, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

namespace raft
{
    namespace linalg
    {

        /** Enum to tell how to compute a norm */
        enum NormType : unsigned short
        {
            /** L0 (actually not a norm): sum((x_i != 0 ? 1 : 0)) */
            L0PseudoNorm = 0,
            /** L1 norm or Manhattan: sum(abs(x_i)) */
            L1Norm = 1,
            /** L2 norm or Euclidean: sqrt(sum(x_i^2)). Note that in some prims the square root is optional,
     in which case it can be specified using a boolean or a functor final_op */
            L2Norm = 2,
            /** Linf norm or Chebyshev: max(abs(x_i)) */
            LinfNorm
        };

    } // namespace linalg
} // namespace raft
