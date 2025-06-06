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

#include "cusolver_wrappers.hpp"

#include <raft/core/resource/cusolver_dn_handle.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cuda_runtime.h>
#else
#include <cuda_runtime_api.h>
#endif

namespace raft
{
    namespace linalg
    {
        namespace detail
        {

            template <typename math_t>
            void eigDC_legacy(raft::resources const& handle,
                              const math_t*          in,
                              std::size_t            n_rows,
                              std::size_t            n_cols,
                              math_t*                eig_vectors,
                              math_t*                eig_vals,
                              cudaStream_t           stream)
            {
                cusolverDnHandle_t cusolverH = resource::get_cusolver_dn_handle(handle);

                int lwork;
                RAFT_CUSOLVER_TRY(cusolverDnsyevd_bufferSize(cusolverH,
                                                             CUSOLVER_EIG_MODE_VECTOR,
                                                             CUBLAS_FILL_MODE_UPPER,
                                                             n_rows,
                                                             in,
                                                             n_cols,
                                                             eig_vals,
                                                             &lwork));

                rmm::device_uvector<math_t> d_work(lwork, stream);
                rmm::device_scalar<int>     d_dev_info(stream);

                raft::matrix::copy(handle,
                                   make_device_matrix_view<const math_t>(in, n_rows, n_cols),
                                   make_device_matrix_view<math_t>(eig_vectors, n_rows, n_cols));

                RAFT_CUSOLVER_TRY(cusolverDnsyevd(cusolverH,
                                                  CUSOLVER_EIG_MODE_VECTOR,
                                                  CUBLAS_FILL_MODE_UPPER,
                                                  n_rows,
                                                  eig_vectors,
                                                  n_cols,
                                                  eig_vals,
                                                  d_work.data(),
                                                  lwork,
                                                  d_dev_info.data(),
                                                  stream));
                RAFT_CUDA_TRY(cudaGetLastError());

                auto dev_info = d_dev_info.value(stream);
                ASSERT(dev_info == 0,
                       "eig.cuh: eigensolver couldn't converge to a solution. "
                       "This usually occurs when some of the features do not vary enough.");
            }

            template <typename math_t>
            void eigDC(raft::resources const& handle,
                       const math_t*          in,
                       std::size_t            n_rows,
                       std::size_t            n_cols,
                       math_t*                eig_vectors,
                       math_t*                eig_vals,
                       cudaStream_t           stream)
            {
#if CUDART_VERSION < 11010
                eigDC_legacy(handle, in, n_rows, n_cols, eig_vectors, eig_vals, stream);
#else
                cusolverDnHandle_t cusolverH = resource::get_cusolver_dn_handle(handle);

                cusolverDnParams_t dn_params = nullptr;
                RAFT_CUSOLVER_TRY(cusolverDnCreateParams(&dn_params));

                size_t workspaceDevice = 0;
                size_t workspaceHost   = 0;
                RAFT_CUSOLVER_TRY(cusolverDnxsyevd_bufferSize(cusolverH,
                                                              dn_params,
                                                              CUSOLVER_EIG_MODE_VECTOR,
                                                              CUBLAS_FILL_MODE_UPPER,
                                                              static_cast<int64_t>(n_rows),
                                                              eig_vectors,
                                                              static_cast<int64_t>(n_cols),
                                                              eig_vals,
                                                              &workspaceDevice,
                                                              &workspaceHost,
                                                              stream));

                rmm::device_uvector<math_t> d_work(workspaceDevice / sizeof(math_t), stream);
                rmm::device_scalar<int>     d_dev_info(stream);
                std::vector<math_t>         h_work(workspaceHost / sizeof(math_t));

                raft::matrix::copy(handle,
                                   make_device_matrix_view<const math_t>(in, n_rows, n_cols),
                                   make_device_matrix_view<math_t>(eig_vectors, n_rows, n_cols));

                RAFT_CUSOLVER_TRY(cusolverDnxsyevd(cusolverH,
                                                   dn_params,
                                                   CUSOLVER_EIG_MODE_VECTOR,
                                                   CUBLAS_FILL_MODE_UPPER,
                                                   static_cast<int64_t>(n_rows),
                                                   eig_vectors,
                                                   static_cast<int64_t>(n_cols),
                                                   eig_vals,
                                                   d_work.data(),
                                                   workspaceDevice,
                                                   h_work.data(),
                                                   workspaceHost,
                                                   d_dev_info.data(),
                                                   stream));

                RAFT_CUDA_TRY(cudaGetLastError());
                RAFT_CUSOLVER_TRY(cusolverDnDestroyParams(dn_params));
                int dev_info = d_dev_info.value(stream);
                ASSERT(dev_info == 0,
                       "eig.cuh: eigensolver couldn't converge to a solution. "
                       "This usually occurs when some of the features do not vary enough.");
#endif
            }

            enum EigVecMemUsage
            {
                OVERWRITE_INPUT,
                COPY_INPUT
            };

            template <typename math_t>
            void eigSelDC(raft::resources const& handle,
                          math_t*                in,
                          std::size_t            n_rows,
                          std::size_t            n_cols,
                          std::size_t            n_eig_vals,
                          math_t*                eig_vectors,
                          math_t*                eig_vals,
                          EigVecMemUsage         memUsage,
                          cudaStream_t           stream)
            {
                cusolverDnHandle_t cusolverH = resource::get_cusolver_dn_handle(handle);

                int lwork;
                int h_meig;

                RAFT_CUSOLVER_TRY(
                    cusolverDnsyevdx_bufferSize(cusolverH,
                                                CUSOLVER_EIG_MODE_VECTOR,
                                                CUSOLVER_EIG_RANGE_I,
                                                CUBLAS_FILL_MODE_UPPER,
                                                static_cast<int64_t>(n_rows),
                                                in,
                                                static_cast<int64_t>(n_cols),
                                                math_t(0.0),
                                                math_t(0.0),
                                                static_cast<int64_t>(n_cols - n_eig_vals + 1),
                                                static_cast<int64_t>(n_cols),
                                                &h_meig,
                                                eig_vals,
                                                &lwork));

                rmm::device_uvector<math_t> d_work(lwork, stream);
                rmm::device_scalar<int>     d_dev_info(stream);
                rmm::device_uvector<math_t> d_eig_vectors(0, stream);

                if(memUsage == OVERWRITE_INPUT)
                {
                    RAFT_CUSOLVER_TRY(
                        cusolverDnsyevdx(cusolverH,
                                         CUSOLVER_EIG_MODE_VECTOR,
                                         CUSOLVER_EIG_RANGE_I,
                                         CUBLAS_FILL_MODE_UPPER,
                                         static_cast<int64_t>(n_rows),
                                         in,
                                         static_cast<int64_t>(n_cols),
                                         math_t(0.0),
                                         math_t(0.0),
                                         static_cast<int64_t>(n_cols - n_eig_vals + 1),
                                         static_cast<int64_t>(n_cols),
                                         &h_meig,
                                         eig_vals,
                                         d_work.data(),
                                         lwork,
                                         d_dev_info.data(),
                                         stream));
                }
                else if(memUsage == COPY_INPUT)
                {
                    d_eig_vectors.resize(n_rows * n_cols, stream);
                    raft::matrix::copy(handle,
                                       make_device_matrix_view<const math_t>(in, n_rows, n_cols),
                                       make_device_matrix_view(eig_vectors, n_rows, n_cols));

                    RAFT_CUSOLVER_TRY(
                        cusolverDnsyevdx(cusolverH,
                                         CUSOLVER_EIG_MODE_VECTOR,
                                         CUSOLVER_EIG_RANGE_I,
                                         CUBLAS_FILL_MODE_UPPER,
                                         static_cast<int64_t>(n_rows),
                                         eig_vectors,
                                         static_cast<int64_t>(n_cols),
                                         math_t(0.0),
                                         math_t(0.0),
                                         static_cast<int64_t>(n_cols - n_eig_vals + 1),
                                         static_cast<int64_t>(n_cols),
                                         &h_meig,
                                         eig_vals,
                                         d_work.data(),
                                         lwork,
                                         d_dev_info.data(),
                                         stream));
                }

                RAFT_CUDA_TRY(cudaGetLastError());

                int dev_info = d_dev_info.value(stream);
                ASSERT(dev_info == 0,
                       "eig.cuh: eigensolver couldn't converge to a solution. "
                       "This usually occurs when some of the features do not vary enough.");

                if(memUsage == OVERWRITE_INPUT)
                {
                    raft::matrix::trunc_zero_origin(
                        handle,
                        make_device_matrix_view<const math_t, size_t, col_major>(
                            in, n_rows, n_eig_vals),
                        make_device_matrix_view<math_t, size_t, col_major>(
                            eig_vectors, n_rows, n_eig_vals));
                }
                else if(memUsage == COPY_INPUT)
                {
                    raft::matrix::trunc_zero_origin(
                        handle,
                        make_device_matrix_view<const math_t, size_t, col_major>(
                            d_eig_vectors.data(), n_rows, n_eig_vals),
                        make_device_matrix_view<math_t, size_t, col_major>(
                            eig_vectors, n_rows, n_eig_vals));
                }
            }

            template <typename math_t>
            void eigJacobi(raft::resources const& handle,
                           const math_t*          in,
                           std::size_t            n_rows,
                           std::size_t            n_cols,
                           math_t*                eig_vectors,
                           math_t*                eig_vals,
                           cudaStream_t           stream,
                           math_t                 tol    = 1.e-7,
                           int                    sweeps = 15)
            {
                cusolverDnHandle_t cusolverH = resource::get_cusolver_dn_handle(handle);

                syevjInfo_t syevj_params = nullptr;
                RAFT_CUSOLVER_TRY(cusolverDnCreateSyevjInfo(&syevj_params));
                RAFT_CUSOLVER_TRY(cusolverDnXsyevjSetTolerance(syevj_params, tol));
                RAFT_CUSOLVER_TRY(cusolverDnXsyevjSetMaxSweeps(syevj_params, sweeps));

                int lwork;
                RAFT_CUSOLVER_TRY(cusolverDnsyevj_bufferSize(cusolverH,
                                                             CUSOLVER_EIG_MODE_VECTOR,
                                                             CUBLAS_FILL_MODE_UPPER,
                                                             static_cast<int64_t>(n_rows),
                                                             eig_vectors,
                                                             static_cast<int64_t>(n_cols),
                                                             eig_vals,
                                                             &lwork,
                                                             syevj_params));

                rmm::device_uvector<math_t> d_work(lwork, stream);
                rmm::device_scalar<int>     dev_info(stream);

                raft::matrix::copy(handle,
                                   make_device_matrix_view<const math_t>(in, n_rows, n_cols),
                                   make_device_matrix_view(eig_vectors, n_rows, n_cols));

                RAFT_CUSOLVER_TRY(cusolverDnsyevj(cusolverH,
                                                  CUSOLVER_EIG_MODE_VECTOR,
                                                  CUBLAS_FILL_MODE_UPPER,
                                                  static_cast<int64_t>(n_rows),
                                                  eig_vectors,
                                                  static_cast<int64_t>(n_cols),
                                                  eig_vals,
                                                  d_work.data(),
                                                  lwork,
                                                  dev_info.data(),
                                                  syevj_params,
                                                  stream));

                int executed_sweeps;
                RAFT_CUSOLVER_TRY(
                    cusolverDnXsyevjGetSweeps(cusolverH, syevj_params, &executed_sweeps));

                RAFT_CUDA_TRY(cudaGetLastError());
                RAFT_CUSOLVER_TRY(cusolverDnDestroySyevjInfo(syevj_params));
            }

        } // namespace detail
    } // namespace linalg
} // namespace raft
