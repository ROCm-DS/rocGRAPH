// Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/vectorized.cuh>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_cooperative_groups.h>
#else
#include <cooperative_groups.h>
#endif

#include <memory>

namespace raft::random
{
    namespace detail
    {

        template <typename Type, typename IntType, typename IdxType, int TPB, bool rowMajor>
        RAFT_KERNEL permuteKernel(
            IntType* perms, Type* out, const Type* in, IdxType a, IdxType b, IdxType N, IdxType D)
        {
            namespace cg                              = cooperative_groups;
            static __device__ constexpr int WARP_SIZE = raft::warp_size();

            int tid = threadIdx.x + blockIdx.x * blockDim.x;

            // having shuffled input indices and coalesced output indices appears
            // to be preferable to the reverse, especially for column major
            IntType inIdx  = ((a * int64_t(tid)) + b) % N;
            IntType outIdx = tid;

            if(perms != nullptr && tid < N)
            {
                perms[outIdx] = inIdx;
            }

            if(out == nullptr || in == nullptr)
            {
                return;
            }

            if(rowMajor)
            {
                cg::thread_block_tile<WARP_SIZE> warp
                    = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());

                __shared__ IntType inIdxShm[TPB];
                __shared__ IntType outIdxShm[TPB];
                inIdxShm[threadIdx.x]  = inIdx;
                outIdxShm[threadIdx.x] = outIdx;
                warp.sync();

                int warpID = threadIdx.x / WARP_SIZE;
                int laneID = threadIdx.x % WARP_SIZE;
                for(int i = warpID * WARP_SIZE; i < warpID * WARP_SIZE + WARP_SIZE; ++i)
                {
                    if(outIdxShm[i] < N)
                    {
#pragma unroll
                        for(int j = laneID; j < D; j += WARP_SIZE)
                        {
                            out[outIdxShm[i] * D + j] = in[inIdxShm[i] * D + j];
                        }
                    }
                }
            }
            else
            {
#pragma unroll
                for(int j = 0; j < D; ++j)
                {
                    if(tid < N)
                    {
                        out[outIdx + j * N] = in[inIdx + j * N];
                    }
                }
            }
        }

        // This is wrapped in a type to allow for partial template specialization
        template <typename Type,
                  typename IntType,
                  typename IdxType,
                  int  TPB,
                  bool rowMajor,
                  int  VLen>
        struct permute_impl_t
        {
            static void permuteImpl(IntType*     perms,
                                    Type*        out,
                                    const Type*  in,
                                    IdxType      N,
                                    IdxType      D,
                                    int          nblks,
                                    IdxType      a,
                                    IdxType      b,
                                    cudaStream_t stream)
            {
                // determine vector type and set new pointers
                typedef typename raft::IOType<Type, VLen>::Type VType;
                VType*       vout = reinterpret_cast<VType*>(out);
                const VType* vin  = reinterpret_cast<const VType*>(in);

                // check if we can execute at this vector length
                if(D % VLen == 0 && raft::is_aligned(vout, sizeof(VType))
                   && raft::is_aligned(vin, sizeof(VType)))
                {
                    permuteKernel<VType, IntType, IdxType, TPB, rowMajor>
                        <<<nblks, TPB, 0, stream>>>(perms, vout, vin, a, b, N, D / VLen);
                    RAFT_CUDA_TRY(cudaPeekAtLastError());
                }
                else
                { // otherwise try the next lower vector length
                    permute_impl_t<Type, IntType, IdxType, TPB, rowMajor, VLen / 2>::permuteImpl(
                        perms, out, in, N, D, nblks, a, b, stream);
                }
            }
        };

        // at vector length 1 we just execute a scalar version to break the recursion
        template <typename Type, typename IntType, typename IdxType, int TPB, bool rowMajor>
        struct permute_impl_t<Type, IntType, IdxType, TPB, rowMajor, 1>
        {
            static void permuteImpl(IntType*     perms,
                                    Type*        out,
                                    const Type*  in,
                                    IdxType      N,
                                    IdxType      D,
                                    int          nblks,
                                    IdxType      a,
                                    IdxType      b,
                                    cudaStream_t stream)
            {
                permuteKernel<Type, IntType, IdxType, TPB, rowMajor>
                    <<<nblks, TPB, 0, stream>>>(perms, out, in, a, b, N, D);
                RAFT_CUDA_TRY(cudaPeekAtLastError());
            }
        };

        template <typename Type, typename IntType = int, typename IdxType = int, int TPB = 256>
        void permute(IntType*     perms,
                     Type*        out,
                     const Type*  in,
                     IntType      D,
                     IntType      N,
                     bool         rowMajor,
                     cudaStream_t stream)
        {
            auto nblks = raft::ceildiv(N, (IntType)TPB);

            // always keep 'a' to be coprime to N
            IdxType a = rand() % N;
            while(raft::gcd(a, N) != 1)
                a = (a + 1) % N;
            IdxType b = rand() % N;

            if(rowMajor)
            {
                permute_impl_t<Type,
                               IntType,
                               IdxType,
                               TPB,
                               true,
                               (16 / sizeof(Type) > 0) ? 16 / sizeof(Type) : 1>::
                    permuteImpl(perms, out, in, N, D, nblks, a, b, stream);
            }
            else
            {
                permute_impl_t<Type, IntType, IdxType, TPB, false, 1>::permuteImpl(
                    perms, out, in, N, D, nblks, a, b, stream);
            }
        }

    }; // end namespace detail
}; // end namespace raft::random
