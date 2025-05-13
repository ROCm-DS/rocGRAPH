/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
 * SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

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

namespace raft
{
    namespace sparse
    {

        /**
 * Quantizes ncols to a valid blockdim, which is
 * a multiple of 32.
 *
 * @param[in] ncols number of blocks to quantize
 */
        template <typename value_idx>
        inline int block_dim(value_idx ncols)
        {
            int blockdim;
            if(ncols <= 32)
                blockdim = 32;
            else if(ncols <= 64)
                blockdim = 64;
            else if(ncols <= 128)
                blockdim = 128;
            else if(ncols <= 256)
                blockdim = 256;
            else if(ncols <= 512)
                blockdim = 512;
            else
                blockdim = 1024;

            return blockdim;
        }

// add similar semantics for __match_any_sync pre-volta (SM_70)
#if __CUDA_ARCH__ < 700
        /**
 * Returns a warp-level mask with 1's for all the threads
 * in the current warp that have the same key.
 * @tparam G
 * @param key
 * @return
 */
        template <typename G>
        __device__ __inline__ unsigned int __match_any_sync(unsigned int init_mask, G key)
        {
            bitmask_type mask       = __ballot_sync(init_mask, true);
            bitmask_type peer_group = 0;
            bool         is_peer;

            do
            {
                // fetch key of first unclaimed lane and compare with this key
                is_peer = (key == __shfl_sync(mask, key, __FFS(mask - 1)));

                // determine which lanes had a match
                peer_group = __ballot_sync(mask, is_peer);

                // remove lanes with matching keys from the pool
                mask = mask ^ peer_group;

                // quit if we had a match
            } while(!is_peer);

            return peer_group;
        }
#endif

        __device__ __inline__ unsigned int get_lowest_peer(bitmask_type peer_group)
        {
            return __FFS(peer_group) - 1;
        }

        template <typename value_idx>
        RAFT_KERNEL iota_fill_block_kernel(value_idx* indices, value_idx ncols)
        {
            int row = blockIdx.x;
            int tid = threadIdx.x;

            for(int i = tid; i < ncols; i += blockDim.x)
            {
                uint64_t idx     = (uint64_t)row * (uint64_t)ncols;
                indices[idx + i] = i;
            }
        }

        template <typename value_idx>
        void iota_fill(value_idx* indices, value_idx nrows, value_idx ncols, cudaStream_t stream)
        {
            int blockdim = block_dim(ncols);

            iota_fill_block_kernel<<<nrows, blockdim, 0, stream>>>(indices, ncols);
        }

        template <typename T>
        __device__ int get_stop_idx(T row, T m, T nnz, const T* ind)
        {
            int stop_idx = 0;
            if(row < (m - 1))
                stop_idx = ind[row + 1];
            else
                stop_idx = nnz;

            return stop_idx;
        }

    }; // namespace sparse
}; // namespace raft
