/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/detail/utils.h>
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
        namespace linalg
        {
            namespace detail
            {

                /**
 * Transpose a set of CSR arrays into a set of CSC arrays.
 * @tparam value_idx : data type of the CSR index arrays
 * @tparam value_t : data type of the CSR data array
 * @param[in] handle : used for invoking cusparse
 * @param[in] csr_indptr : CSR row index array
 * @param[in] csr_indices : CSR column indices array
 * @param[in] csr_data : CSR data array
 * @param[out] csc_indptr : CSC row index array
 * @param[out] csc_indices : CSC column indices array
 * @param[out] csc_data : CSC data array
 * @param[in] csr_nrows : Number of rows in CSR
 * @param[in] csr_ncols : Number of columns in CSR
 * @param[in] nnz : Number of nonzeros of CSR
 * @param[in] stream : Cuda stream for ordering events
 */
                template <typename value_idx, typename value_t>
                void csr_transpose(cusparseHandle_t handle,
                                   const value_idx* csr_indptr,
                                   const value_idx* csr_indices,
                                   const value_t*   csr_data,
                                   value_idx*       csc_indptr,
                                   value_idx*       csc_indices,
                                   value_t*         csc_data,
                                   value_idx        csr_nrows,
                                   value_idx        csr_ncols,
                                   value_idx        nnz,
                                   cudaStream_t     stream)
                {
                    size_t convert_csc_workspace_size = 0;

                    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecsr2csc_bufferSize(
                        handle,
                        csr_nrows,
                        csr_ncols,
                        nnz,
                        csr_data,
                        csr_indptr,
                        csr_indices,
                        csc_data,
                        csc_indptr,
                        csc_indices,
                        CUSPARSE_ACTION_NUMERIC,
                        CUSPARSE_INDEX_BASE_ZERO,
                        CUSPARSE_CSR2CSC_ALG1,
                        &convert_csc_workspace_size,
                        stream));

                    rmm::device_uvector<char> convert_csc_workspace(convert_csc_workspace_size,
                                                                    stream);

                    RAFT_CUSPARSE_TRY(
                        raft::sparse::detail::cusparsecsr2csc(handle,
                                                              csr_nrows,
                                                              csr_ncols,
                                                              nnz,
                                                              csr_data,
                                                              csr_indptr,
                                                              csr_indices,
                                                              csc_data,
                                                              csc_indptr,
                                                              csc_indices,
                                                              CUSPARSE_ACTION_NUMERIC,
                                                              CUSPARSE_INDEX_BASE_ZERO,
                                                              CUSPARSE_CSR2CSC_ALG1,
                                                              convert_csc_workspace.data(),
                                                              stream));
                }

            }; // end NAMESPACE detail
        }; // end NAMESPACE linalg
    }; // end NAMESPACE sparse
}; // end NAMESPACE raft
