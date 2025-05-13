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

#include <algorithm>
#include <iostream>

namespace raft
{
    namespace sparse
    {
        namespace convert
        {
            namespace detail
            {

                template <typename value_idx = int, int TPB_X = 32>
                RAFT_KERNEL csr_to_coo_kernel(const value_idx* row_ind,
                                              value_idx        m,
                                              value_idx*       coo_rows,
                                              value_idx        nnz)
                {
                    // row-based matrix 1 thread per row
                    value_idx row = (blockIdx.x * TPB_X) + threadIdx.x;
                    if(row < m)
                    {
                        value_idx start_idx = row_ind[row];
                        value_idx stop_idx  = get_stop_idx(row, m, nnz, row_ind);
                        for(value_idx i = start_idx; i < stop_idx; i++)
                            coo_rows[i] = row;
                    }
                }

                /**
 * @brief Convert a CSR row_ind array to a COO rows array
 * @param row_ind: Input CSR row_ind array
 * @param m: size of row_ind array
 * @param coo_rows: Output COO row array
 * @param nnz: size of output COO row array
 * @param stream: cuda stream to use
 */
                template <typename value_idx = int, int TPB_X = 32>
                void csr_to_coo(const value_idx* row_ind,
                                value_idx        m,
                                value_idx*       coo_rows,
                                value_idx        nnz,
                                cudaStream_t     stream)
                {
                    // @TODO: Use cusparse for this.
                    dim3 grid(raft::ceildiv(m, (value_idx)TPB_X), 1, 1);
                    dim3 blk(TPB_X, 1, 1);

                    csr_to_coo_kernel<value_idx, TPB_X>
                        <<<grid, blk, 0, stream>>>(row_ind, m, coo_rows, nnz);

                    RAFT_CUDA_TRY(cudaGetLastError());
                }

            }; // end NAMESPACE detail
        }; // end NAMESPACE convert
    }; // end NAMESPACE sparse
}; // end NAMESPACE raft
