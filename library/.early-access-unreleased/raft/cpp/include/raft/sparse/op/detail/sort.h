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

#include <raft/sparse/coo.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/detail/utils.h>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/exec_policy.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cuda_runtime.h>

#include <hipsparse/hipsparse.h>
#else
#include <cuda_runtime.h>

#include <cusparse_v2.h>
#endif
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <algorithm>

namespace raft
{
    namespace sparse
    {
        namespace op
        {
            namespace detail
            {

                struct TupleComp
                {
                    template <typename one, typename two>
                    __host__ __device__

                        bool
                        operator()(const one& t1, const two& t2)
                    {
                        // sort first by each sample's color,
                        if(thrust::get<0>(t1) < thrust::get<0>(t2))
                            return true;
                        if(thrust::get<0>(t1) > thrust::get<0>(t2))
                            return false;

                        // then sort by value in descending order
                        return thrust::get<1>(t1) < thrust::get<1>(t2);
                    }
                };

                /**
 * @brief Sorts the arrays that comprise the coo matrix
 * by row and then by column.
 *
 * @param m number of rows in coo matrix
 * @param n number of cols in coo matrix
 * @param nnz number of non-zeros
 * @param rows rows array from coo matrix
 * @param cols cols array from coo matrix
 * @param vals vals array from coo matrix
 * @param stream: cuda stream to use
 */
                template <typename T>
                void coo_sort(
                    int m, int n, int nnz, int* rows, int* cols, T* vals, cudaStream_t stream)
                {
                    auto coo_indices = thrust::make_zip_iterator(thrust::make_tuple(rows, cols));

                    // get all the colors in contiguous locations so we can map them to warps.
                    thrust::sort_by_key(rmm::exec_policy(stream),
                                        coo_indices,
                                        coo_indices + nnz,
                                        vals,
                                        TupleComp());
                }

                /**
 * @brief Sort the underlying COO arrays by row
 * @tparam T: the type name of the underlying value array
 * @param in: COO to sort by row
 * @param stream: the cuda stream to use
 */
                template <typename T>
                void coo_sort(COO<T>* const in, cudaStream_t stream)
                {
                    coo_sort<T>(in->n_rows,
                                in->n_cols,
                                in->nnz,
                                in->rows(),
                                in->cols(),
                                in->vals(),
                                stream);
                }

                /**
 * Sorts a COO by its weight
 * @tparam value_idx
 * @tparam value_t
 * @param[inout] rows source edges
 * @param[inout] cols dest edges
 * @param[inout] data edge weights
 * @param[in] nnz number of edges in edge list
 * @param[in] stream cuda stream for which to order cuda operations
 */
                template <typename value_idx, typename value_t>
                void coo_sort_by_weight(value_idx*   rows,
                                        value_idx*   cols,
                                        value_t*     data,
                                        value_idx    nnz,
                                        cudaStream_t stream)
                {
                    thrust::device_ptr<value_t> t_data = thrust::device_pointer_cast(data);

                    auto first = thrust::make_zip_iterator(thrust::make_tuple(rows, cols));

                    thrust::sort_by_key(rmm::exec_policy(stream), t_data, t_data + nnz, first);
                }
            }; // namespace detail
        }; // namespace op
    }; // end NAMESPACE sparse
}; // end NAMESPACE raft
