// Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#pragma once

#ifdef __HIP_PLATFORM_AMD__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include <raft/util/cuda_utils.cuh>

namespace raft
{
    namespace linalg
    {
        namespace detail
        {

            template <int warpSize, int rpb>
            struct NormalizeThinPolicy
            {
                static constexpr int LogicalWarpSize = warpSize;
                static constexpr int RowsPerBlock    = rpb;
                static constexpr int ThreadsPerBlock = LogicalWarpSize * RowsPerBlock;
            };

            template <typename Policy,
                      typename Type,
                      typename IdxType,
                      typename MainLambda,
                      typename ReduceLambda,
                      typename FinalLambda>
            RAFT_KERNEL __launch_bounds__(Policy::ThreadsPerBlock)
                coalesced_normalize_thin_kernel(Type*        out,
                                                const Type*  in,
                                                IdxType      D,
                                                IdxType      N,
                                                Type         init,
                                                MainLambda   main_op,
                                                ReduceLambda reduce_op,
                                                FinalLambda  fin_op,
                                                Type         eps)
            {
                IdxType i = threadIdx.y + (Policy::RowsPerBlock * static_cast<IdxType>(blockIdx.x));
                if(i >= N)
                    return;

                Type acc = init;
                for(IdxType j = threadIdx.x; j < D; j += Policy::LogicalWarpSize)
                {
                    Type val = in[j + D * i];
                    acc      = reduce_op(acc, main_op(val, j));
                }
                acc = raft::logicalWarpReduce<Policy::LogicalWarpSize>(acc, reduce_op);
                acc = fin_op(acc);
                if(acc <= eps)
                    return;
                for(IdxType j = threadIdx.x; j < D; j += Policy::LogicalWarpSize)
                {
                    out[j + D * i] = in[j + D * i] / acc;
                }
            }

            template <typename Policy,
                      typename Type,
                      typename IdxType,
                      typename MainLambda,
                      typename ReduceLambda,
                      typename FinalLambda>
            inline void coalesced_normalize_thin(Type*        out,
                                                 const Type*  in,
                                                 IdxType      D,
                                                 IdxType      N,
                                                 Type         init,
                                                 cudaStream_t stream,
                                                 MainLambda   main_op,
                                                 ReduceLambda reduce_op,
                                                 FinalLambda  fin_op,
                                                 Type         eps)
            {
                dim3 grid(ceildiv(N, (IdxType)Policy::RowsPerBlock), 1, 1);
                dim3 block(Policy::LogicalWarpSize, Policy::RowsPerBlock, 1);
                coalesced_normalize_thin_kernel<Policy><<<grid, block, 0, stream>>>(
                    out, in, D, N, init, main_op, reduce_op, fin_op, eps);
                RAFT_CUDA_TRY(cudaPeekAtLastError());
            }

            template <int TPB,
                      typename Type,
                      typename IdxType,
                      typename MainLambda,
                      typename ReduceLambda,
                      typename FinalLambda>
            RAFT_KERNEL __launch_bounds__(TPB)
                coalesced_normalize_medium_kernel(Type*        out,
                                                  const Type*  in,
                                                  IdxType      D,
                                                  IdxType      N,
                                                  Type         init,
                                                  MainLambda   main_op,
                                                  ReduceLambda reduce_op,
                                                  FinalLambda  fin_op,
                                                  Type         eps)
            {
                typedef cub::BlockReduce<Type, TPB, cub::BLOCK_REDUCE_RAKING> BlockReduce;
                __shared__ typename BlockReduce::TempStorage                  temp_storage;
                __shared__ Type                                               bcast_acc;
                Type                                                          thread_data = init;
                IdxType rowStart = blockIdx.x * D;
                for(IdxType i = threadIdx.x; i < D; i += TPB)
                {
                    IdxType idx = rowStart + i;
                    thread_data = reduce_op(thread_data, main_op(in[idx], i));
                }
                Type acc = BlockReduce(temp_storage).Reduce(thread_data, reduce_op);
                if(threadIdx.x == 0)
                {
                    bcast_acc = fin_op(acc);
                }
                __syncthreads();
                if(bcast_acc <= eps)
                    return;
                for(IdxType i = threadIdx.x; i < D; i += TPB)
                {
                    IdxType idx = rowStart + i;
                    out[idx]    = in[idx] / bcast_acc;
                }
            }

            template <int TPB,
                      typename Type,
                      typename IdxType,
                      typename MainLambda,
                      typename ReduceLambda,
                      typename FinalLambda>
            inline void coalesced_normalize_medium(Type*        out,
                                                   const Type*  in,
                                                   IdxType      D,
                                                   IdxType      N,
                                                   Type         init,
                                                   cudaStream_t stream,
                                                   MainLambda   main_op,
                                                   ReduceLambda reduce_op,
                                                   FinalLambda  fin_op,
                                                   Type         eps)
            {
                coalesced_normalize_medium_kernel<TPB>
                    <<<N, TPB, 0, stream>>>(out, in, D, N, init, main_op, reduce_op, fin_op, eps);
                RAFT_CUDA_TRY(cudaPeekAtLastError());
            }

            template <typename Type,
                      typename IdxType,
                      typename MainLambda,
                      typename ReduceLambda,
                      typename FinalLambda>
            void coalesced_normalize(Type*        out,
                                     const Type*  in,
                                     IdxType      D,
                                     IdxType      N,
                                     Type         init,
                                     cudaStream_t stream,
                                     MainLambda   main_op,
                                     ReduceLambda reduce_op,
                                     FinalLambda  fin_op,
                                     Type         eps)
            {
                const IdxType numSMs = raft::getMultiProcessorCount();
                if(D <= IdxType(256) || (D <= IdxType(512) && N >= 4 * numSMs))
                {
                    if(D <= IdxType(2))
                    {
                        coalesced_normalize_thin<NormalizeThinPolicy<2, 64>>(
                            out, in, D, N, init, stream, main_op, reduce_op, fin_op, eps);
                    }
                    else if(D <= IdxType(4))
                    {
                        coalesced_normalize_thin<NormalizeThinPolicy<4, 32>>(
                            out, in, D, N, init, stream, main_op, reduce_op, fin_op, eps);
                    }
                    else if(D <= IdxType(8))
                    {
                        coalesced_normalize_thin<NormalizeThinPolicy<8, 16>>(
                            out, in, D, N, init, stream, main_op, reduce_op, fin_op, eps);
                    }
                    else if(D <= IdxType(16))
                    {
                        coalesced_normalize_thin<NormalizeThinPolicy<16, 8>>(
                            out, in, D, N, init, stream, main_op, reduce_op, fin_op, eps);
                    }
                    else
                    {
                        coalesced_normalize_thin<NormalizeThinPolicy<32, 4>>(
                            out, in, D, N, init, stream, main_op, reduce_op, fin_op, eps);
                    }
                }
                else
                {
                    coalesced_normalize_medium<256>(
                        out, in, D, N, init, stream, main_op, reduce_op, fin_op, eps);
                }
            }

        } // namespace detail
    } // namespace linalg
} // namespace raft
