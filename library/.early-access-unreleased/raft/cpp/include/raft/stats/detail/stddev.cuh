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
#pragma once

#include <raft/linalg/binary_op.cuh>
#include <raft/util/cuda_utils.cuh>

#ifdef __HIP_PLATFORM_AMD__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#endif

namespace raft
{
    namespace stats
    {
        namespace detail
        {

            ///@todo: ColPerBlk has been tested only for 32!
            template <typename Type, typename IdxType, int TPB, int ColsPerBlk = 32>
            RAFT_KERNEL stddevKernelRowMajor(Type* std, const Type* data, IdxType D, IdxType N)
            {
                const int     RowsPerBlkPerIter = TPB / ColsPerBlk;
                IdxType       thisColId         = threadIdx.x % ColsPerBlk;
                IdxType       thisRowId         = threadIdx.x / ColsPerBlk;
                IdxType       colId             = thisColId + ((IdxType)blockIdx.y * ColsPerBlk);
                IdxType       rowId       = thisRowId + ((IdxType)blockIdx.x * RowsPerBlkPerIter);
                Type          thread_data = Type(0);
                const IdxType stride      = RowsPerBlkPerIter * gridDim.x;
                for(IdxType i = rowId; i < N; i += stride)
                {
                    Type val = (colId < D) ? data[i * D + colId] : Type(0);
                    thread_data += val * val;
                }
                __shared__ Type sstd[ColsPerBlk];
                if(threadIdx.x < ColsPerBlk)
                    sstd[threadIdx.x] = Type(0);
                __syncthreads();
                raft::myAtomicAdd(sstd + thisColId, thread_data);
                __syncthreads();
                if(threadIdx.x < ColsPerBlk && colId < D)
                    raft::myAtomicAdd(std + colId, sstd[thisColId]);
            }

            template <typename Type, typename IdxType, int TPB>
            RAFT_KERNEL stddevKernelColMajor(
                Type* std, const Type* data, const Type* mu, IdxType D, IdxType N)
            {
                typedef cub::BlockReduce<Type, TPB>          BlockReduce;
                __shared__ typename BlockReduce::TempStorage temp_storage;
                Type                                         thread_data = Type(0);
                IdxType                                      colStart    = N * blockIdx.x;
                Type                                         m           = mu[blockIdx.x];
                for(IdxType i = threadIdx.x; i < N; i += TPB)
                {
                    IdxType idx  = colStart + i;
                    Type    diff = data[idx] - m;
                    thread_data += diff * diff;
                }
                Type acc = BlockReduce(temp_storage).Sum(thread_data);
                if(threadIdx.x == 0)
                {
                    std[blockIdx.x] = raft::sqrt(acc / N);
                }
            }

            template <typename Type, typename IdxType, int TPB>
            RAFT_KERNEL varsKernelColMajor(
                Type* var, const Type* data, const Type* mu, IdxType D, IdxType N)
            {
                typedef cub::BlockReduce<Type, TPB>          BlockReduce;
                __shared__ typename BlockReduce::TempStorage temp_storage;
                Type                                         thread_data = Type(0);
                IdxType                                      colStart    = N * blockIdx.x;
                Type                                         m           = mu[blockIdx.x];
                for(IdxType i = threadIdx.x; i < N; i += TPB)
                {
                    IdxType idx  = colStart + i;
                    Type    diff = data[idx] - m;
                    thread_data += diff * diff;
                }
                Type acc = BlockReduce(temp_storage).Sum(thread_data);
                if(threadIdx.x == 0)
                {
                    var[blockIdx.x] = acc / N;
                }
            }

            /**
 * @brief Compute stddev of the input matrix
 *
 * Stddev operation is assumed to be performed on a given column.
 *
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @param std the output stddev vector
 * @param data the input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param sample whether to evaluate sample stddev or not. In other words,
 * whether
 *  to normalize the output using N-1 or N, for true or false, respectively
 * @param rowMajor whether the input data is row or col major
 * @param stream cuda stream where to launch work
 */
            template <typename Type, typename IdxType = int>
            void stddev(Type*        std,
                        const Type*  data,
                        const Type*  mu,
                        IdxType      D,
                        IdxType      N,
                        bool         sample,
                        bool         rowMajor,
                        cudaStream_t stream)
            {
                static const int TPB = 256;
                if(rowMajor)
                {
                    static const int RowsPerThread = 4;
                    static const int ColsPerBlk    = 32;
                    static const int RowsPerBlk    = (TPB / ColsPerBlk) * RowsPerThread;
                    dim3             grid(raft::ceildiv(N, (IdxType)RowsPerBlk),
                              raft::ceildiv(D, (IdxType)ColsPerBlk));
                    RAFT_CUDA_TRY(cudaMemset(std, 0, sizeof(Type) * D));
                    stddevKernelRowMajor<Type, IdxType, TPB, ColsPerBlk>
                        <<<grid, TPB, 0, stream>>>(std, data, D, N);
                    Type ratio = Type(1) / (sample ? Type(N - 1) : Type(N));
                    raft::linalg::binaryOp(
                        std,
                        std,
                        mu,
                        D,
                        [ratio] __device__(Type a, Type b) {
                            return raft::sqrt(a * ratio - b * b);
                        },
                        stream);
                }
                else
                {
                    stddevKernelColMajor<Type, IdxType, TPB>
                        <<<D, TPB, 0, stream>>>(std, data, mu, D, N);
                }
                RAFT_CUDA_TRY(cudaPeekAtLastError());
            }

            /**
 * @brief Compute variance of the input matrix
 *
 * Variance operation is assumed to be performed on a given column.
 *
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @param var the output stddev vector
 * @param data the input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param sample whether to evaluate sample stddev or not. In other words,
 * whether
 *  to normalize the output using N-1 or N, for true or false, respectively
 * @param rowMajor whether the input data is row or col major
 * @param stream cuda stream where to launch work
 */
            template <typename Type, typename IdxType = int>
            void vars(Type*        var,
                      const Type*  data,
                      const Type*  mu,
                      IdxType      D,
                      IdxType      N,
                      bool         sample,
                      bool         rowMajor,
                      cudaStream_t stream)
            {
                static const int TPB = 256;
                if(rowMajor)
                {
                    static const int RowsPerThread = 4;
                    static const int ColsPerBlk    = 32;
                    static const int RowsPerBlk    = (TPB / ColsPerBlk) * RowsPerThread;
                    dim3             grid(raft::ceildiv(N, (IdxType)RowsPerBlk),
                              raft::ceildiv(D, (IdxType)ColsPerBlk));
                    RAFT_CUDA_TRY(cudaMemset(var, 0, sizeof(Type) * D));
                    stddevKernelRowMajor<Type, IdxType, TPB, ColsPerBlk>
                        <<<grid, TPB, 0, stream>>>(var, data, D, N);
                    Type ratio = Type(1) / (sample ? Type(N - 1) : Type(N));
                    raft::linalg::binaryOp(
                        var,
                        var,
                        mu,
                        D,
                        [ratio] __device__(Type a, Type b) { return a * ratio - b * b; },
                        stream);
                }
                else
                {
                    varsKernelColMajor<Type, IdxType, TPB>
                        <<<D, TPB, 0, stream>>>(var, data, mu, D, N);
                }
                RAFT_CUDA_TRY(cudaPeekAtLastError());
            }

        } // namespace detail
    } // namespace stats
} // namespace raft
