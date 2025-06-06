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
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#endif

#include <algorithm>

namespace raft
{
    namespace label
    {
        namespace detail
        {

            /**
 * Get unique class labels.
 *
 * The y array is assumed to store class labels. The unique values are selected
 * from this array.
 *
 * \tparam value_t numeric type of the arrays with class labels
 * \param [in] y device array of labels, size [n]
 * \param [in] n number of labels
 * \param [out] unique device array of unique labels, unallocated on entry,
 *   on exit it has size [n_unique]
 * \param [out] n_unique number of unique labels
 * \param [in] stream cuda stream
 */
            template <typename value_t>
            int getUniquelabels(rmm::device_uvector<value_t>& unique,
                                value_t*                      y,
                                size_t                        n,
                                cudaStream_t                  stream)
            {
                rmm::device_scalar<int>      d_num_selected(stream);
                rmm::device_uvector<value_t> workspace(n, stream);
                size_t                       bytes  = 0;
                size_t                       bytes2 = 0;

                // Query how much temporary storage we will need for cub operations
                // and allocate it
                cub::DeviceRadixSort::SortKeys(
                    NULL, bytes, y, workspace.data(), n, 0, sizeof(value_t) * 8, stream);
                cub::DeviceSelect::Unique(NULL,
                                          bytes2,
                                          workspace.data(),
                                          workspace.data(),
                                          d_num_selected.data(),
                                          n,
                                          stream);
                bytes = std::max(bytes, bytes2);
                rmm::device_uvector<char> cub_storage(bytes, stream);

                // Select Unique classes
                cub::DeviceRadixSort::SortKeys(cub_storage.data(),
                                               bytes,
                                               y,
                                               workspace.data(),
                                               n,
                                               0,
                                               sizeof(value_t) * 8,
                                               stream);
                cub::DeviceSelect::Unique(cub_storage.data(),
                                          bytes,
                                          workspace.data(),
                                          workspace.data(),
                                          d_num_selected.data(),
                                          n,
                                          stream);

                int n_unique = d_num_selected.value(stream);
                // Copy unique classes to output
                unique.resize(n_unique, stream);
                raft::copy(unique.data(), workspace.data(), n_unique, stream);

                return n_unique;
            }

            /**
 * Assign one versus rest labels.
 *
 * The output labels will have values +/-1:
 * y_out = (y == y_unique[idx]) ? +1 : -1;
 *
 * The output type currently is set to value_t, but for SVM in principle we are
 * free to choose other type for y_out (it should represent +/-1, and it is used
 * in floating point arithmetics).
 *
 * \param [in] y device array if input labels, size [n]
 * \param [in] n number of labels
 * \param [in] y_unique device array of unique labels, size [n_classes]
 * \param [in] n_classes number of unique labels
 * \param [out] y_out device array of output labels
 * \param [in] idx index of unique label that should be labeled as 1
 * \param [in] stream cuda stream
 */
            template <typename value_t>
            void getOvrlabels(value_t*     y,
                              int          n,
                              value_t*     y_unique,
                              int          n_classes,
                              value_t*     y_out,
                              int          idx,
                              cudaStream_t stream)
            {
                ASSERT(idx < n_classes,
                       "Parameter idx should not be larger than the number "
                       "of classes");
                raft::linalg::unaryOp(
                    y_out,
                    y,
                    n,
                    [idx, y_unique] __device__(value_t y) { return y == y_unique[idx] ? +1 : -1; },
                    stream);
                RAFT_CUDA_TRY(cudaPeekAtLastError());
            }

            // TODO: add one-versus-one selection: select two classes, relabel them to
            // +/-1, return array with the new class labels and corresponding indices.

            template <typename Type, int TPB_X, typename Lambda>
            RAFT_KERNEL map_label_kernel(Type*  map_ids,
                                         size_t N_labels,
                                         Type*  in,
                                         Type*  out,
                                         size_t N,
                                         Lambda filter_op,
                                         bool   zero_based = false)
            {
                int tid = threadIdx.x + blockIdx.x * TPB_X;
                if(tid < N)
                {
                    if(!filter_op(in[tid]))
                    {
                        for(size_t i = 0; i < N_labels; i++)
                        {
                            if(in[tid] == map_ids[i])
                            {
                                out[tid] = i + !zero_based;
                                break;
                            }
                        }
                    }
                }
            }

            /**
 * Maps an input array containing a series of numbers into a new array
 * where numbers have been mapped to a monotonically increasing set
 * of labels. This can be useful in machine learning algorithms, for instance,
 * where a given set of labels is not taken from a monotonically increasing
 * set. This can happen if they are filtered or if only a subset of the
 * total labels are used in a dataset. This is also useful in graph algorithms
 * where a set of vertices need to be labeled in a monotonically increasing
 * order.
 * @tparam Type the numeric type of the input and output arrays
 * @tparam Lambda the type of an optional filter function, which determines
 * which items in the array to map.
 * @param out the output monotonic array
 * @param in input label array
 * @param N number of elements in the input array
 * @param stream cuda stream to use
 * @param filter_op an optional function for specifying which values
 * should have monotonically increasing labels applied to them.
 */
            template <typename Type, typename Lambda>
            void make_monotonic(Type*        out,
                                Type*        in,
                                size_t       N,
                                cudaStream_t stream,
                                Lambda       filter_op,
                                bool         zero_based = false)
            {
                static const size_t TPB_X = 256;

                dim3 blocks(raft::ceildiv(N, TPB_X));
                dim3 threads(TPB_X);

                rmm::device_uvector<Type> map_ids(0, stream);
                int                       num_clusters = getUniquelabels(map_ids, in, N, stream);

                map_label_kernel<Type, TPB_X><<<blocks, threads, 0, stream>>>(
                    map_ids.data(), num_clusters, in, out, N, filter_op, zero_based);
            }

            /**
 * Maps an input array containing a series of numbers into a new array
 * where numbers have been mapped to a monotonically increasing set
 * of labels. This can be useful in machine learning algorithms, for instance,
 * where a given set of labels is not taken from a monotonically increasing
 * set. This can happen if they are filtered or if only a subset of the
 * total labels are used in a dataset. This is also useful in graph algorithms
 * where a set of vertices need to be labeled in a monotonically increasing
 * order.
 * @tparam Type the numeric type of the input and output arrays
 * @tparam Lambda the type of an optional filter function, which determines
 * which items in the array to map.
 * @param out output label array with labels assigned monotonically
 * @param in input label array
 * @param N number of elements in the input array
 * @param stream cuda stream to use
 */
            template <typename Type>
            void make_monotonic(
                Type* out, Type* in, size_t N, cudaStream_t stream, bool zero_based = false)
            {
                make_monotonic<Type>(out, in, N, stream, raft::const_op(false), zero_based);
            }

        }; // namespace detail
    }; // namespace label
}; // end namespace raft
