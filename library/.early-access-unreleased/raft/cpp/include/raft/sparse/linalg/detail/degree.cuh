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

#include <raft/sparse/coo.hpp>
#include <raft/sparse/detail/utils.h>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/device_atomics.cuh>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include <stdio.h>

namespace raft
{
    namespace sparse
    {
        namespace linalg
        {
            namespace detail
            {

                /**
 * @brief Count all the rows in the coo row array and place them in the
 * results matrix, indexed by row.
 *
 * @tparam TPB_X: number of threads to use per block
 * @param rows the rows array of the coo matrix
 * @param nnz the size of the rows array
 * @param results array to place results
 */
                template <int TPB_X = 64, typename T = int>
                RAFT_KERNEL coo_degree_kernel(const T* rows, int nnz, T* results)
                {
                    int row = (blockIdx.x * TPB_X) + threadIdx.x;
                    if(row < nnz)
                    {
                        atomicAdd(results + rows[row], (T)1);
                    }
                }

                /**
 * @brief Count the number of values for each row
 * @tparam TPB_X: number of threads to use per block
 * @param rows: rows array of the COO matrix
 * @param nnz: size of the rows array
 * @param results: output result array
 * @param stream: cuda stream to use
 */
                template <int TPB_X = 64, typename T = int>
                void coo_degree(const T* rows, int nnz, T* results, cudaStream_t stream)
                {
                    dim3 grid_rc(raft::ceildiv(nnz, TPB_X), 1, 1);
                    dim3 blk_rc(TPB_X, 1, 1);

                    coo_degree_kernel<TPB_X><<<grid_rc, blk_rc, 0, stream>>>(rows, nnz, results);
                    RAFT_CUDA_TRY(cudaGetLastError());
                }

                template <int TPB_X = 64, typename T>
                RAFT_KERNEL
                    coo_degree_nz_kernel(const int* rows, const T* vals, int nnz, int* results)
                {
                    int row = (blockIdx.x * TPB_X) + threadIdx.x;
                    if(row < nnz && vals[row] != 0.0)
                    {
                        raft::myAtomicAdd(results + rows[row], 1);
                    }
                }

                template <int TPB_X = 64, typename T>
                RAFT_KERNEL coo_degree_scalar_kernel(
                    const int* rows, const T* vals, int nnz, T scalar, int* results)
                {
                    int row = (blockIdx.x * TPB_X) + threadIdx.x;
                    if(row < nnz && vals[row] != scalar)
                    {
                        raft::myAtomicAdd(results + rows[row], 1);
                    }
                }

                /**
 * @brief Count the number of values for each row that doesn't match a particular scalar
 * @tparam TPB_X: number of threads to use per block
 * @tparam T: the type name of the underlying value arrays
 * @param rows: Input COO row array
 * @param vals: Input COO val arrays
 * @param nnz: size of input COO arrays
 * @param scalar: scalar to match for counting rows
 * @param results: output row counts
 * @param stream: cuda stream to use
 */
                template <int TPB_X = 64, typename T>
                void coo_degree_scalar(const int*   rows,
                                       const T*     vals,
                                       int          nnz,
                                       T            scalar,
                                       int*         results,
                                       cudaStream_t stream = 0)
                {
                    dim3 grid_rc(raft::ceildiv(nnz, TPB_X), 1, 1);
                    dim3 blk_rc(TPB_X, 1, 1);
                    coo_degree_scalar_kernel<TPB_X, T>
                        <<<grid_rc, blk_rc, 0, stream>>>(rows, vals, nnz, scalar, results);
                }

                /**
 * @brief Count the number of nonzeros for each row
 * @tparam TPB_X: number of threads to use per block
 * @tparam T: the type name of the underlying value arrays
 * @param rows: Input COO row array
 * @param vals: Input COO val arrays
 * @param nnz: size of input COO arrays
 * @param results: output row counts
 * @param stream: cuda stream to use
 */
                template <int TPB_X = 64, typename T>
                void coo_degree_nz(
                    const int* rows, const T* vals, int nnz, int* results, cudaStream_t stream)
                {
                    dim3 grid_rc(raft::ceildiv(nnz, TPB_X), 1, 1);
                    dim3 blk_rc(TPB_X, 1, 1);
                    coo_degree_nz_kernel<TPB_X, T>
                        <<<grid_rc, blk_rc, 0, stream>>>(rows, vals, nnz, results);
                }

            }; // end NAMESPACE detail
        }; // end NAMESPACE linalg
    }; // end NAMESPACE sparse
}; // end NAMESPACE raft
