// Copyright (c) 2022, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <raft/core/operators.hpp>
#include <raft/linalg/coalesced_reduction.cuh>
#include <raft/linalg/strided_reduction.cuh>

namespace raft
{
    namespace linalg
    {
        namespace detail
        {

            template <typename InType,
                      typename OutType      = InType,
                      typename IdxType      = int,
                      typename MainLambda   = raft::identity_op,
                      typename ReduceLambda = raft::add_op,
                      typename FinalLambda  = raft::identity_op>
            void reduce(OutType*      dots,
                        const InType* data,
                        IdxType       D,
                        IdxType       N,
                        OutType       init,
                        bool          rowMajor,
                        bool          alongRows,
                        cudaStream_t  stream,
                        bool          inplace   = false,
                        MainLambda    main_op   = raft::identity_op(),
                        ReduceLambda  reduce_op = raft::add_op(),
                        FinalLambda   final_op  = raft::identity_op())
            {
                if(rowMajor && alongRows)
                {
                    raft::linalg::coalescedReduction<InType, OutType, IdxType>(
                        dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
                }
                else if(rowMajor && !alongRows)
                {
                    raft::linalg::stridedReduction<InType, OutType, IdxType>(
                        dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
                }
                else if(!rowMajor && alongRows)
                {
                    raft::linalg::stridedReduction<InType, OutType, IdxType>(
                        dots, data, N, D, init, stream, inplace, main_op, reduce_op, final_op);
                }
                else
                {
                    raft::linalg::coalescedReduction<InType, OutType, IdxType>(
                        dots, data, N, D, init, stream, inplace, main_op, reduce_op, final_op);
                }
            }

        }; // end namespace detail
    }; // end namespace linalg
}; // end namespace raft
