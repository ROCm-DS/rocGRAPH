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

#include "cublas_wrappers.hpp"

#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resources.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cublas_v2.h>
#else
#include <cublas_v2.h>
#endif

namespace raft
{
    namespace linalg
    {
        namespace detail
        {

            template <typename math_t, bool DevicePointerMode = false>
            void gemv(raft::resources const& handle,
                      const bool             trans_a,
                      const int              m,
                      const int              n,
                      const math_t*          alpha,
                      const math_t*          A,
                      const int              lda,
                      const math_t*          x,
                      const int              incx,
                      const math_t*          beta,
                      math_t*                y,
                      const int              incy,
                      cudaStream_t           stream)
            {
                cublasHandle_t cublas_h = resource::get_cublas_handle(handle);
                detail::cublas_device_pointer_mode<DevicePointerMode> pmode(cublas_h);
                RAFT_CUBLAS_TRY(detail::cublasgemv(cublas_h,
                                                   trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                                                   m,
                                                   n,
                                                   alpha,
                                                   A,
                                                   lda,
                                                   x,
                                                   incx,
                                                   beta,
                                                   y,
                                                   incy,
                                                   stream));
            }

            template <typename math_t>
            void gemv(raft::resources const& handle,
                      const math_t*          A,
                      const int              n_rows,
                      const int              n_cols,
                      const math_t*          x,
                      const int              incx,
                      math_t*                y,
                      const int              incy,
                      const bool             trans_a,
                      const math_t           alpha,
                      const math_t           beta,
                      cudaStream_t           stream)
            {
                gemv(handle,
                     trans_a,
                     n_rows,
                     n_cols,
                     &alpha,
                     A,
                     n_rows,
                     x,
                     incx,
                     &beta,
                     y,
                     incy,
                     stream);
            }

            template <typename math_t>
            void gemv(raft::resources const& handle,
                      const math_t*          A,
                      const int              n_rows_a,
                      const int              n_cols_a,
                      const math_t*          x,
                      math_t*                y,
                      const bool             trans_a,
                      const math_t           alpha,
                      const math_t           beta,
                      cudaStream_t           stream)
            {
                gemv(handle, A, n_rows_a, n_cols_a, x, 1, y, 1, trans_a, alpha, beta, stream);
            }

            template <typename math_t>
            void gemv(raft::resources const& handle,
                      const math_t*          A,
                      const int              n_rows_a,
                      const int              n_cols_a,
                      const math_t*          x,
                      math_t*                y,
                      const bool             trans_a,
                      cudaStream_t           stream)
            {
                math_t alpha = math_t(1);
                math_t beta  = math_t(0);

                gemv(handle, A, n_rows_a, n_cols_a, x, 1, y, 1, trans_a, alpha, beta, stream);
            }

            template <typename math_t>
            void gemv(raft::resources const& handle,
                      const math_t*          A,
                      const int              n_rows_a,
                      const int              n_cols_a,
                      const int              lda,
                      const math_t*          x,
                      math_t*                y,
                      const bool             trans_a,
                      const math_t           alpha,
                      const math_t           beta,
                      cudaStream_t           stream)
            {
                cublasHandle_t    cublas_h = resource::get_cublas_handle(handle);
                cublasOperation_t op_a     = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
                RAFT_CUBLAS_TRY(cublasgemv(
                    cublas_h, op_a, n_rows_a, n_cols_a, &alpha, A, lda, x, 1, &beta, y, 1, stream));
            }

            template <typename math_t>
            void gemv(raft::resources const& handle,
                      const math_t*          A,
                      const int              n_rows_a,
                      const int              n_cols_a,
                      const int              lda,
                      const math_t*          x,
                      math_t*                y,
                      const bool             trans_a,
                      cudaStream_t           stream)
            {
                math_t alpha = math_t(1);
                math_t beta  = math_t(0);
                gemv(handle, A, n_rows_a, n_cols_a, lda, x, y, trans_a, alpha, beta, stream);
            }

        }; // namespace detail
    }; // namespace linalg
}; // namespace raft
