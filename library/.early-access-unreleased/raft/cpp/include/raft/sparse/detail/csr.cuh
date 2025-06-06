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

#include <rmm/device_scalar.hpp>
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
        namespace detail
        {

            //@TODO: Pull this out into a separate file

            struct WeakCCState
            {
            public:
                bool* m;

                WeakCCState(bool* m)
                    : m(m)
                {
                }
            };

            template <typename Index_, int TPB_X = 256, typename Lambda>
            RAFT_KERNEL weak_cc_label_device(Index_* __restrict__ labels,
                                             const Index_* __restrict__ row_ind,
                                             const Index_* __restrict__ row_ind_ptr,
                                             Index_ nnz,
                                             bool* __restrict__ m,
                                             Index_ start_vertex_id,
                                             Index_ batch_size,
                                             Index_ N,
                                             Lambda filter_op)
            {
                Index_ tid       = threadIdx.x + blockIdx.x * TPB_X;
                Index_ global_id = tid + start_vertex_id;
                if(tid < batch_size && global_id < N)
                {
                    Index_ start = __ldg(row_ind + tid);

                    Index_ ci, cj;
                    bool   ci_mod      = false;
                    ci                 = labels[global_id];
                    bool ci_allow_prop = filter_op(global_id);

                    Index_ end = get_stop_idx(tid, batch_size, nnz, row_ind);
                    /// TODO: add one element to row_ind and avoid get_stop_idx
                    for(Index_ j = start; j < end; j++)
                    {
                        Index_ j_ind       = __ldg(row_ind_ptr + j);
                        cj                 = labels[j_ind];
                        bool cj_allow_prop = filter_op(j_ind);
                        if(ci < cj && ci_allow_prop)
                        {
                            if(sizeof(Index_) == 4)
                                atomicMin((int*)(labels + j_ind), ci);
                            else if(sizeof(Index_) == 8)
                                atomicMin((long long int*)(labels + j_ind), ci);
                            if(cj_allow_prop)
                                *m = true;
                        }
                        else if(ci > cj && cj_allow_prop)
                        {
                            ci     = cj;
                            ci_mod = true;
                        }
                    }
                    if(ci_mod)
                    {
                        if(sizeof(Index_) == 4)
                            atomicMin((int*)(labels + global_id), ci);
                        else if(sizeof(Index_) == 8)
                            atomicMin((long long int*)(labels + global_id), ci);
                        if(ci_allow_prop)
                            *m = true;
                    }
                }
            }

            template <typename Index_, int TPB_X = 256, typename Lambda>
            RAFT_KERNEL weak_cc_init_all_kernel(Index_* labels,
                                                Index_  N,
                                                Index_  MAX_LABEL,
                                                Lambda  filter_op)
            {
                Index_ tid = threadIdx.x + blockIdx.x * TPB_X;
                if(tid < N)
                {
                    if(filter_op(tid))
                        labels[tid] = tid + 1;
                    else
                        labels[tid] = MAX_LABEL;
                }
            } // namespace sparse

            /**
 * @brief Partial calculation of the weakly connected components in the
 * context of a batched algorithm: the labels are computed wrt the sub-graph
 * represented by the given CSR matrix of dimensions batch_size * N.
 * Note that this overwrites the labels array and it is the responsibility of
 * the caller to combine the results from different batches
 * (cf label/merge_labels.cuh)
 *
 * @tparam Index_ the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @param labels an array for the output labels
 * @param row_ind the compressed row index of the CSR array
 * @param row_ind_ptr the row index pointer of the CSR array
 * @param nnz the size of row_ind_ptr array
 * @param N number of vertices
 * @param start_vertex_id the starting vertex index for the current batch
 * @param batch_size number of vertices for current batch
 * @param state instance of inter-batch state management
 * @param stream the cuda stream to use
 * @param filter_op an optional filtering function to determine which points
 * should get considered for labeling. It gets global indexes (not batch-wide!)
 */
            template <typename Index_, int TPB_X = 256, typename Lambda = auto(Index_)->bool>
            void weak_cc_batched(Index_*       labels,
                                 const Index_* row_ind,
                                 const Index_* row_ind_ptr,
                                 Index_        nnz,
                                 Index_        N,
                                 Index_        start_vertex_id,
                                 Index_        batch_size,
                                 WeakCCState*  state,
                                 cudaStream_t  stream,
                                 Lambda        filter_op)
            {
                ASSERT(sizeof(Index_) == 4 || sizeof(Index_) == 8, "Index_ should be 4 or 8 bytes");

                bool host_m;

                Index_ MAX_LABEL = std::numeric_limits<Index_>::max();
                weak_cc_init_all_kernel<Index_, TPB_X>
                    <<<raft::ceildiv(N, Index_(TPB_X)), TPB_X, 0, stream>>>(
                        labels, N, MAX_LABEL, filter_op);
                RAFT_CUDA_TRY(cudaPeekAtLastError());

                int n_iters = 0;
                do
                {
                    RAFT_CUDA_TRY(cudaMemsetAsync(state->m, false, sizeof(bool), stream));

                    weak_cc_label_device<Index_, TPB_X>
                        <<<raft::ceildiv(batch_size, Index_(TPB_X)), TPB_X, 0, stream>>>(
                            labels,
                            row_ind,
                            row_ind_ptr,
                            nnz,
                            state->m,
                            start_vertex_id,
                            batch_size,
                            N,
                            filter_op);
                    RAFT_CUDA_TRY(cudaPeekAtLastError());

                    //** Updating m *
                    raft::update_host(&host_m, state->m, 1, stream);
                    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

                    n_iters++;
                } while(host_m);
            }

        }; // namespace detail
    }; // namespace sparse
}; // namespace raft
