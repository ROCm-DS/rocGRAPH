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

#include <raft/core/operators.hpp>
#include <raft/linalg/unary_op.cuh>
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

                /**
 * Slice consecutive rows from a CSR array and populate newly sliced indptr array
 * @tparam value_idx
 * @param[in] start_row : beginning row to slice
 * @param[in] stop_row : ending row to slice
 * @param[in] indptr : indptr of input CSR to slice
 * @param[out] indptr_out : output sliced indptr to populate
 * @param[in] start_offset : beginning column offset of input indptr
 * @param[in] stop_offset : ending column offset of input indptr
 * @param[in] stream : cuda stream for ordering events
 */
                template <typename value_idx>
                void csr_row_slice_indptr(value_idx        start_row,
                                          value_idx        stop_row,
                                          const value_idx* indptr,
                                          value_idx*       indptr_out,
                                          value_idx*       start_offset,
                                          value_idx*       stop_offset,
                                          cudaStream_t     stream)
                {
                    raft::update_host(start_offset, indptr + start_row, 1, stream);
                    raft::update_host(stop_offset, indptr + stop_row + 1, 1, stream);

                    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

                    value_idx s_offset = *start_offset;

                    // 0-based indexing so we need to add 1 to stop row. Because we want n_rows+1,
                    // we add another 1 to stop row.
                    raft::copy_async(
                        indptr_out, indptr + start_row, (stop_row + 2) - start_row, stream);

                    raft::linalg::unaryOp<value_idx>(indptr_out,
                                                     indptr_out,
                                                     (stop_row + 2) - start_row,
                                                     raft::sub_const_op<value_idx>(s_offset),
                                                     stream);
                }

                /**
 * Slice rows from a CSR, populate column and data arrays
 * @tparam[in] value_idx : data type of CSR index arrays
 * @tparam[in] value_t : data type of CSR data array
 * @param[in] start_offset : beginning column offset to slice
 * @param[in] stop_offset : ending column offset to slice
 * @param[in] indices : column indices array from input CSR
 * @param[in] data : data array from input CSR
 * @param[out] indices_out : output column indices array
 * @param[out] data_out : output data array
 * @param[in] stream : cuda stream for ordering events
 */
                template <typename value_idx, typename value_t>
                void csr_row_slice_populate(value_idx        start_offset,
                                            value_idx        stop_offset,
                                            const value_idx* indices,
                                            const value_t*   data,
                                            value_idx*       indices_out,
                                            value_t*         data_out,
                                            cudaStream_t     stream)
                {
                    raft::copy(
                        indices_out, indices + start_offset, stop_offset - start_offset, stream);
                    raft::copy(data_out, data + start_offset, stop_offset - start_offset, stream);
                }

            }; // namespace detail
        }; // namespace op
    }; // end NAMESPACE sparse
}; // end NAMESPACE raft
