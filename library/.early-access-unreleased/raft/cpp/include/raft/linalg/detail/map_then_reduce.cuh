// Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/core/resources.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/vectorized.cuh>

#ifdef __HIP_PLATFORM_AMD__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#endif

namespace raft
{
    namespace linalg
    {
        namespace detail
        {

            struct sum_tag
            {
            };

            template <typename InType, typename OutType, int TPB>
            __device__ void reduce(OutType* out, const InType acc, sum_tag)
            {
                typedef cub::BlockReduce<InType, TPB>        BlockReduce;
                __shared__ typename BlockReduce::TempStorage temp_storage;
                OutType tmp = BlockReduce(temp_storage).Sum(acc);
                if(threadIdx.x == 0)
                {
                    raft::myAtomicAdd(out, tmp);
                }
            }

            template <typename InType, typename OutType, int TPB, typename ReduceLambda>
            __device__ void reduce(OutType* out, const InType acc, ReduceLambda op)
            {
                typedef cub::BlockReduce<InType, TPB>        BlockReduce;
                __shared__ typename BlockReduce::TempStorage temp_storage;
                OutType tmp = BlockReduce(temp_storage).Reduce(acc, op);
                if(threadIdx.x == 0)
                {
                    raft::myAtomicReduce(out, tmp, op);
                }
            }

            template <typename InType,
                      typename OutType,
                      typename IdxType,
                      typename MapOp,
                      typename ReduceLambda,
                      int TPB,
                      typename... Args>
            RAFT_KERNEL mapThenReduceKernel(OutType*      out,
                                            IdxType       len,
                                            OutType       neutral,
                                            MapOp         map,
                                            ReduceLambda  op,
                                            const InType* in,
                                            Args... args)
            {
                OutType acc = neutral;
                auto    idx = (threadIdx.x + (blockIdx.x * blockDim.x));

                if(idx < len)
                {
                    acc = map(in[idx], args[idx]...);
                }

                __syncthreads();

                reduce<InType, OutType, TPB>(out, acc, op);
            }

            template <typename InType,
                      typename OutType,
                      typename IdxType,
                      typename MapOp,
                      typename ReduceLambda,
                      int TPB,
                      typename... Args>
            void mapThenReduceImpl(OutType*      out,
                                   IdxType       len,
                                   OutType       neutral,
                                   MapOp         map,
                                   ReduceLambda  op,
                                   cudaStream_t  stream,
                                   const InType* in,
                                   Args... args)
            {
                raft::update_device(out, &neutral, 1, stream);
                const int nblks = raft::ceildiv(len, IdxType(TPB));
                mapThenReduceKernel<InType, OutType, IdxType, MapOp, ReduceLambda, TPB, Args...>
                    <<<nblks, TPB, 0, stream>>>(out, len, neutral, map, op, in, args...);
                RAFT_CUDA_TRY(cudaPeekAtLastError());
            }

        }; // end namespace detail
    }; // end namespace linalg
}; // end namespace raft
