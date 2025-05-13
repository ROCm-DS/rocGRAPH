// Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/core/device_resources.hpp>
#include <raft/matrix/select_k_types.hpp>
#include <raft/util/raft_explicit.hpp> // RAFT_EXPLICIT

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif // __half

#include <cstdint> // uint32_t

#ifdef RAFT_EXPLICIT_INSTANTIATE_ONLY

namespace raft::matrix::detail
{

    template <typename T, typename IdxT>
    void select_k(raft::resources const& handle,
                  const T*               in_val,
                  const IdxT*            in_idx,
                  size_t                 batch_size,
                  size_t                 len,
                  int                    k,
                  T*                     out_val,
                  IdxT*                  out_idx,
                  bool                   select_min,
                  bool                   sorted = false,
                  SelectAlgo             algo   = SelectAlgo::kAuto,
                  const IdxT*            len_i  = nullptr) RAFT_EXPLICIT;
} // namespace raft::matrix::detail

#endif // RAFT_EXPLICIT_INSTANTIATE_ONLY

#define instantiate_raft_matrix_detail_select_k(T, IdxT)                                     \
    extern template void raft::matrix::detail::select_k(raft::resources const&   handle,     \
                                                        const T*                 in_val,     \
                                                        const IdxT*              in_idx,     \
                                                        size_t                   batch_size, \
                                                        size_t                   len,        \
                                                        int                      k,          \
                                                        T*                       out_val,    \
                                                        IdxT*                    out_idx,    \
                                                        bool                     select_min, \
                                                        bool                     sorted,     \
                                                        raft::matrix::SelectAlgo algo,       \
                                                        const IdxT*              len_i)
instantiate_raft_matrix_detail_select_k(__half, uint32_t);
instantiate_raft_matrix_detail_select_k(__half, int64_t);
instantiate_raft_matrix_detail_select_k(float, int64_t);
instantiate_raft_matrix_detail_select_k(float, uint32_t);
// needed for brute force knn
instantiate_raft_matrix_detail_select_k(float, int);
// We did not have these two for double before, but there are tests for them. We
// therefore include them here.
instantiate_raft_matrix_detail_select_k(double, int64_t);
instantiate_raft_matrix_detail_select_k(double, uint32_t);

#undef instantiate_raft_matrix_detail_select_k
