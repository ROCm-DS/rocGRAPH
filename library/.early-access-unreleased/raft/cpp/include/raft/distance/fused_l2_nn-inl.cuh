// Copyright (c) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/*
 * Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#ifndef __FUSED_L2_NN_H
#define __FUSED_L2_NN_H

#pragma once

#include <raft/core/resources.hpp>
#include <raft/distance/detail/fused_distance_nn/fused_l2_nn.cuh>
#include <raft/distance/fused_distance_nn_helpers.cuh>
#include <raft/linalg/contractions.cuh>
#include <raft/util/cuda_utils.cuh>

#ifdef __HIP_PLATFORM_AMD__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#endif

#include <stdint.h>

#include <limits>
#include <type_traits>

namespace raft
{
    namespace distance
    {

        /**
 * \ingroup fused_l2_nn
 * @{
 */
        /**
 * @brief Fused L2 distance and 1-nearest-neighbor computation in a single call.
 *
 * The benefits of such a call are 2-fold: 1) eliminate the need for an
 * intermediate buffer to store the output of gemm 2) reduce the memory read
 * traffic on this intermediate buffer, otherwise needed during the reduction
 * phase for 1-NN.
 *
 * @tparam DataT     data type
 * @tparam OutT      output type to either store 1-NN indices and their minimum
 *                   distances or store only the min distances. Accordingly, one
 *                   has to pass an appropriate `ReduceOpT`
 * @tparam IdxT      indexing arithmetic type
 * @tparam ReduceOpT A struct to perform the final needed reduction operation
 *                   and also to initialize the output array elements with the
 *                   appropriate initial value needed for reduction.
 *
 * @param[out] min           will contain the reduced output (Length = `m`)
 *                           (on device)
 * @param[in]  x             first matrix. Row major. Dim = `m x k`.
 *                           (on device).
 * @param[in]  y             second matrix. Row major. Dim = `n x k`.
 *                           (on device).
 * @param[in]  xn            L2 squared norm of `x`. Length = `m`. (on device).
 * @param[in]  yn            L2 squared norm of `y`. Length = `n`. (on device)
 * @param[in]  m             gemm m
 * @param[in]  n             gemm n
 * @param[in]  k             gemm k
 * @param[in]  workspace     temp workspace. Size = sizeof(int)*m. (on device)
 * @param[in]  redOp         reduction operator in the epilogue
 * @param[in] pairRedOp reduction operation on key value pairs
 * @param[in]  sqrt          Whether the output `minDist` should contain L2-sqrt
 * @param[in]  initOutBuffer whether to initialize the output buffer before the
 *                           main kernel launch
 * @param[in]  stream        cuda stream
 */
        template <typename DataT,
                  typename OutT,
                  typename IdxT,
                  typename ReduceOpT,
                  typename KVPReduceOpT>
        void fusedL2NN(OutT*        min,
                       const DataT* x,
                       const DataT* y,
                       const DataT* xn,
                       const DataT* yn,
                       IdxT         m,
                       IdxT         n,
                       IdxT         k,
                       void*        workspace,
                       ReduceOpT    redOp,
                       KVPReduceOpT pairRedOp,
                       bool         sqrt,
                       bool         initOutBuffer,
                       cudaStream_t stream)
        {
            // When k is smaller than 32, the Policy4x4 results in redundant calculations
            // as it uses tiles that have k=32. Therefore, use a "skinny" policy instead
            // that uses tiles with a smaller value of k.
            bool is_skinny = k < 32;

            size_t bytes = sizeof(DataT) * k;
            auto   px    = reinterpret_cast<uintptr_t>(x);
            auto   py    = reinterpret_cast<uintptr_t>(y);
            if(16 % sizeof(DataT) == 0 && bytes % 16 == 0 && px % 16 == 0 && py % 16 == 0)
            {
                if(is_skinny)
                {
                    detail::fusedL2NNImpl<
                        DataT,
                        OutT,
                        IdxT,
                        typename linalg::Policy4x4Skinny<DataT, 16 / sizeof(DataT)>::Policy,
                        ReduceOpT>(min,
                                   x,
                                   y,
                                   xn,
                                   yn,
                                   m,
                                   n,
                                   k,
                                   (int*)workspace,
                                   redOp,
                                   pairRedOp,
                                   sqrt,
                                   initOutBuffer,
                                   stream);
                }
                else
                {
                    detail::fusedL2NNImpl<
                        DataT,
                        OutT,
                        IdxT,
                        typename linalg::Policy4x4<DataT, 16 / sizeof(DataT)>::Policy,
                        ReduceOpT>(min,
                                   x,
                                   y,
                                   xn,
                                   yn,
                                   m,
                                   n,
                                   k,
                                   (int*)workspace,
                                   redOp,
                                   pairRedOp,
                                   sqrt,
                                   initOutBuffer,
                                   stream);
                }
            }
            else if(8 % sizeof(DataT) == 0 && bytes % 8 == 0 && px % 8 == 0 && py % 8 == 0)
            {
                if(is_skinny)
                {
                    detail::fusedL2NNImpl<
                        DataT,
                        OutT,
                        IdxT,
                        typename linalg::Policy4x4Skinny<DataT, 8 / sizeof(DataT)>::Policy,
                        ReduceOpT>(min,
                                   x,
                                   y,
                                   xn,
                                   yn,
                                   m,
                                   n,
                                   k,
                                   (int*)workspace,
                                   redOp,
                                   pairRedOp,
                                   sqrt,
                                   initOutBuffer,
                                   stream);
                }
                else
                {
                    detail::fusedL2NNImpl<
                        DataT,
                        OutT,
                        IdxT,
                        typename linalg::Policy4x4<DataT, 8 / sizeof(DataT)>::Policy,
                        ReduceOpT>(min,
                                   x,
                                   y,
                                   xn,
                                   yn,
                                   m,
                                   n,
                                   k,
                                   (int*)workspace,
                                   redOp,
                                   pairRedOp,
                                   sqrt,
                                   initOutBuffer,
                                   stream);
                }
            }
            else
            {
                if(is_skinny)
                {
                    detail::fusedL2NNImpl<DataT,
                                          OutT,
                                          IdxT,
                                          typename linalg::Policy4x4Skinny<DataT, 1>::Policy,
                                          ReduceOpT>(min,
                                                     x,
                                                     y,
                                                     xn,
                                                     yn,
                                                     m,
                                                     n,
                                                     k,
                                                     (int*)workspace,
                                                     redOp,
                                                     pairRedOp,
                                                     sqrt,
                                                     initOutBuffer,
                                                     stream);
                }
                else
                {
                    detail::fusedL2NNImpl<DataT,
                                          OutT,
                                          IdxT,
                                          typename linalg::Policy4x4<DataT, 1>::Policy,
                                          ReduceOpT>(min,
                                                     x,
                                                     y,
                                                     xn,
                                                     yn,
                                                     m,
                                                     n,
                                                     k,
                                                     (int*)workspace,
                                                     redOp,
                                                     pairRedOp,
                                                     sqrt,
                                                     initOutBuffer,
                                                     stream);
                }
            }
        }

        /**
 * @brief Wrapper around fusedL2NN with minimum reduction operators.
 *
 * fusedL2NN cannot be compiled in the distance library due to the lambda
 * operators, so this wrapper covers the most common case (minimum).
 * This should be preferred to the more generic API when possible, in order to
 * reduce compilation times for users of the shared library.
 *
 * @tparam DataT     data type
 * @tparam OutT      output type to either store 1-NN indices and their minimum
 *                   distances (e.g. raft::KeyValuePair<int, float>) or store only the min
 * distances.
 * @tparam IdxT      indexing arithmetic type
 * @param[out] min           will contain the reduced output (Length = `m`)
 *                           (on device)
 * @param[in]  x             first matrix. Row major. Dim = `m x k`.
 *                           (on device).
 * @param[in]  y             second matrix. Row major. Dim = `n x k`.
 *                           (on device).
 * @param[in]  xn            L2 squared norm of `x`. Length = `m`. (on device).
 * @param[in]  yn            L2 squared norm of `y`. Length = `n`. (on device)
 * @param[in]  m             gemm m
 * @param[in]  n             gemm n
 * @param[in]  k             gemm k
 * @param[in]  workspace     temp workspace. Size = sizeof(int)*m. (on device)
 * @param[in]  sqrt          Whether the output `minDist` should contain L2-sqrt
 * @param[in]  initOutBuffer whether to initialize the output buffer before the
 *                           main kernel launch
 * @param[in]  stream        cuda stream
 */
        template <typename DataT, typename OutT, typename IdxT>
        void fusedL2NNMinReduce(OutT*        min,
                                const DataT* x,
                                const DataT* y,
                                const DataT* xn,
                                const DataT* yn,
                                IdxT         m,
                                IdxT         n,
                                IdxT         k,
                                void*        workspace,
                                bool         sqrt,
                                bool         initOutBuffer,
                                cudaStream_t stream)
        {
            MinAndDistanceReduceOp<IdxT, DataT> redOp;
            KVPMinReduce<IdxT, DataT>           pairRedOp;

            fusedL2NN<DataT, OutT, IdxT>(min,
                                         x,
                                         y,
                                         xn,
                                         yn,
                                         m,
                                         n,
                                         k,
                                         workspace,
                                         redOp,
                                         pairRedOp,
                                         sqrt,
                                         initOutBuffer,
                                         stream);
        }

        /** @} */

    } // namespace distance
} // namespace raft

#endif
