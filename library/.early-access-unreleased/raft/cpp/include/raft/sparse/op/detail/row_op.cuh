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

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/detail/utils.h>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cuda_runtime.h>
#include <raft/cusparse.h>
#else
#include <cuda_runtime.h>

#include <cusparse_v2.h>
#endif
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <stdio.h>

#include <algorithm>
#include <iostream>

namespace raft
{
    namespace sparse
    {
        namespace op
        {
            namespace detail
            {

                template <typename T, int TPB_X = 256, typename Lambda = auto(T, T, T)->void>
                RAFT_KERNEL csr_row_op_kernel(const T* row_ind, T n_rows, T nnz, Lambda op)
                {
                    T row = blockIdx.x * TPB_X + threadIdx.x;
                    if(row < n_rows)
                    {
                        T start_idx = row_ind[row];
                        T stop_idx  = row < n_rows - 1 ? row_ind[row + 1] : nnz;
                        op(row, start_idx, stop_idx);
                    }
                }

                /**
 * @brief Perform a custom row operation on a CSR matrix in batches.
 * @tparam T numerical type of row_ind array
 * @tparam TPB_X number of threads per block to use for underlying kernel
 * @tparam Lambda type of custom operation function
 * @param row_ind the CSR row_ind array to perform parallel operations over
 * @param n_rows total number vertices in graph
 * @param nnz number of non-zeros
 * @param op custom row operation functor accepting the row and beginning index.
 * @param stream cuda stream to use
 */
                template <typename Index_,
                          int TPB_X       = 256,
                          typename Lambda = auto(Index_, Index_, Index_)->void>
                void csr_row_op(const Index_* row_ind,
                                Index_        n_rows,
                                Index_        nnz,
                                Lambda        op,
                                cudaStream_t  stream)
                {
                    dim3 grid(raft::ceildiv(n_rows, Index_(TPB_X)), 1, 1);
                    dim3 blk(TPB_X, 1, 1);
                    csr_row_op_kernel<Index_, TPB_X>
                        <<<grid, blk, 0, stream>>>(row_ind, n_rows, nnz, op);

                    RAFT_CUDA_TRY(cudaPeekAtLastError());
                }

            }; // namespace detail
        }; // namespace op
    }; // end NAMESPACE sparse
}; // end NAMESPACE raft
