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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cusparse_handle.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/detail/utils.h>
#include <raft/sparse/linalg/degree.cuh>
#include <raft/sparse/op/row_op.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

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
        namespace convert
        {
            namespace detail
            {

                template <typename value_t>
                void coo_to_csr(raft::resources const& handle,
                                const int*             srcRows,
                                const int*             srcCols,
                                const value_t*         srcVals,
                                int                    nnz,
                                int                    m,
                                int*                   dst_offsets,
                                int*                   dstCols,
                                value_t*               dstVals)
                {
                    auto                     stream         = resource::get_cuda_stream(handle);
                    auto                     cusparseHandle = resource::get_cusparse_handle(handle);
                    rmm::device_uvector<int> dstRows(nnz, stream);
                    RAFT_CUDA_TRY(cudaMemcpyAsync(dstRows.data(),
                                                  srcRows,
                                                  sizeof(int) * nnz,
                                                  cudaMemcpyDeviceToDevice,
                                                  stream));
                    RAFT_CUDA_TRY(cudaMemcpyAsync(
                        dstCols, srcCols, sizeof(int) * nnz, cudaMemcpyDeviceToDevice, stream));
                    auto buffSize = raft::sparse::detail::cusparsecoosort_bufferSizeExt(
                        cusparseHandle, m, m, nnz, srcRows, srcCols, stream);
                    rmm::device_uvector<char> pBuffer(buffSize, stream);
                    rmm::device_uvector<int>  P(nnz, stream);
                    RAFT_CUSPARSE_TRY(
                        cusparseCreateIdentityPermutation(cusparseHandle, nnz, P.data()));
                    raft::sparse::detail::cusparsecoosortByRow(cusparseHandle,
                                                               m,
                                                               m,
                                                               nnz,
                                                               dstRows.data(),
                                                               dstCols,
                                                               P.data(),
                                                               pBuffer.data(),
                                                               stream);
                    raft::sparse::detail::cusparsegthr(
                        cusparseHandle, nnz, srcVals, dstVals, P.data(), stream);
                    raft::sparse::detail::cusparsecoo2csr(
                        cusparseHandle, dstRows.data(), nnz, m, dst_offsets, stream);
                    RAFT_CUDA_TRY(cudaDeviceSynchronize());
                }

                /**
 * @brief Generate the row indices array for a sorted COO matrix
 *
 * @param rows: COO rows array
 * @param nnz: size of COO rows array
 * @param row_ind: output row indices array
 * @param m: number of rows in dense matrix
 * @param stream: cuda stream to use
 */
                template <typename T>
                void sorted_coo_to_csr(
                    const T* rows, int nnz, T* row_ind, int m, cudaStream_t stream)
                {
                    rmm::device_uvector<T> row_counts(m, stream);

                    RAFT_CUDA_TRY(cudaMemsetAsync(row_counts.data(), 0, m * sizeof(T), stream));

                    linalg::coo_degree(rows, nnz, row_counts.data(), stream);

                    // create csr compressed row index from row counts
                    thrust::device_ptr<T> row_counts_d
                        = thrust::device_pointer_cast(row_counts.data());
                    thrust::device_ptr<T> c_ind_d = thrust::device_pointer_cast(row_ind);
                    exclusive_scan(
                        rmm::exec_policy(stream), row_counts_d, row_counts_d + m, c_ind_d);
                }

            }; // end NAMESPACE detail
        }; // end NAMESPACE convert
    }; // end NAMESPACE sparse
}; // end NAMESPACE raft
