// Copyright (c) 2022, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <raft/core/operators.hpp>
#include <raft/linalg/norm_types.hpp>
#include <raft/linalg/reduce.cuh>

namespace raft
{
    namespace linalg
    {
        namespace detail
        {

            template <typename Type, typename IdxType, typename Lambda>
            void rowNormCaller(Type*        dots,
                               const Type*  data,
                               IdxType      D,
                               IdxType      N,
                               NormType     type,
                               bool         rowMajor,
                               cudaStream_t stream,
                               Lambda       fin_op)
            {
                switch(type)
                {
                case L1Norm:
                    raft::linalg::reduce<Type, Type, IdxType>(dots,
                                                              data,
                                                              D,
                                                              N,
                                                              (Type)0,
                                                              rowMajor,
                                                              true,
                                                              stream,
                                                              false,
                                                              raft::abs_op(),
                                                              raft::add_op(),
                                                              fin_op);
                    break;
                case L2Norm:
                    raft::linalg::reduce<Type, Type, IdxType>(dots,
                                                              data,
                                                              D,
                                                              N,
                                                              (Type)0,
                                                              rowMajor,
                                                              true,
                                                              stream,
                                                              false,
                                                              raft::sq_op(),
                                                              raft::add_op(),
                                                              fin_op);
                    break;
                case LinfNorm:
                    raft::linalg::reduce<Type, Type, IdxType>(dots,
                                                              data,
                                                              D,
                                                              N,
                                                              (Type)0,
                                                              rowMajor,
                                                              true,
                                                              stream,
                                                              false,
                                                              raft::abs_op(),
                                                              raft::max_op(),
                                                              fin_op);
                    break;
                default:
                    THROW("Unsupported norm type: %d", type);
                };
            }

            template <typename Type, typename IdxType, typename Lambda>
            void colNormCaller(Type*        dots,
                               const Type*  data,
                               IdxType      D,
                               IdxType      N,
                               NormType     type,
                               bool         rowMajor,
                               cudaStream_t stream,
                               Lambda       fin_op)
            {
                switch(type)
                {
                case L1Norm:
                    raft::linalg::reduce<Type, Type, IdxType>(dots,
                                                              data,
                                                              D,
                                                              N,
                                                              (Type)0,
                                                              rowMajor,
                                                              false,
                                                              stream,
                                                              false,
                                                              raft::abs_op(),
                                                              raft::add_op(),
                                                              fin_op);
                    break;
                case L2Norm:
                    raft::linalg::reduce<Type, Type, IdxType>(dots,
                                                              data,
                                                              D,
                                                              N,
                                                              (Type)0,
                                                              rowMajor,
                                                              false,
                                                              stream,
                                                              false,
                                                              raft::sq_op(),
                                                              raft::add_op(),
                                                              fin_op);
                    break;
                case LinfNorm:
                    raft::linalg::reduce<Type, Type, IdxType>(dots,
                                                              data,
                                                              D,
                                                              N,
                                                              (Type)0,
                                                              rowMajor,
                                                              false,
                                                              stream,
                                                              false,
                                                              raft::abs_op(),
                                                              raft::max_op(),
                                                              fin_op);
                    break;
                default:
                    THROW("Unsupported norm type: %d", type);
                };
            }

        }; // end namespace detail
    }; // end namespace linalg
}; // end namespace raft
