/*
 * SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <hipsolver/hipsolver.h>

#define CUSOLVERAPI

// types
#define csrqrInfo hipsolvercsrqrInfo
#define csrqrInfo_t hipsolvercsrqrInfo_t
#define cusolverDnHandle_t hipsolverDnHandle_t
#define cusolverDnParams_t hipsolverDnParams_t
#define cusolverEigMode_t hipsolverEigMode_t
#define cusolverEigRange_t hipsolverEigRange_t
#define cusolverSpHandle_t hipsolverSpHandle_t
#define cusolverStatus_t hipsolverStatus_t
#define cusparseMatDescr_t hipsparseMatDescr_t
#define gesvdjInfo_t hipsolverGesvdjInfo_t
#define syevjInfo hipsyevjInfo
#define syevjInfo_t hipsolverSyevjInfo_t

// macros/constants
#define CUSOLVER_EIG_MODE_VECTOR HIPSOLVER_EIG_MODE_VECTOR
#define CUSOLVER_EIG_RANGE_I HIPSOLVER_EIG_RANGE_I
#define CUSOLVER_STATUS_ALLOC_FAILED HIPSOLVER_STATUS_ALLOC_FAILED
#define CUSOLVER_STATUS_ARCH_MISMATCH HIPSOLVER_STATUS_ARCH_MISMATCH
#define CUSOLVER_STATUS_EXECUTION_FAILED HIPSOLVER_STATUS_EXECUTION_FAILED
#define CUSOLVER_STATUS_INTERNAL_ERROR HIPSOLVER_STATUS_INTERNAL_ERROR
#define CUSOLVER_STATUS_INVALID_VALUE HIPSOLVER_STATUS_INVALID_VALUE
#define CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED
#define CUSOLVER_STATUS_NOT_INITIALIZED HIPSOLVER_STATUS_NOT_INITIALIZED
#define CUSOLVER_STATUS_NOT_SUPPORTED HIPSOLVER_STATUS_NOT_SUPPORTED
#define CUSOLVER_STATUS_SUCCESS HIPSOLVER_STATUS_SUCCESS
#define CUSOLVER_STATUS_ZERO_PIVOT HIPSOLVER_STATUS_ZERO_PIVOT

// functions
#define cusolverDnCreate hipsolverDnCreate
#define cusolverDnCreateGesvdjInfo hipsolverDnCreateGesvdjInfo
#define cusolverDnCreateParams hipsolverDnCreateParams
#define cusolverDnCreateSyevjInfo hipsolverDnCreateSyevjInfo
#define cusolverDnDestroy hipsolverDnDestroy
#define cusolverDnDestroyGesvdjInfo hipsolverDnDestroyGesvdjInfo
#define cusolverDnDestroyParams hipsolverDnDestroyParams
#define cusolverDnDestroySyevjInfo hipsolverDnDestroySyevjInfo
#define cusolverDnDgeqrf hipsolverDnDgeqrf
#define cusolverDnDgeqrf_bufferSize hipsolverDnDgeqrf_bufferSize
#define cusolverDnDgesvd hipsolverDnDgesvd
#define cusolverDnDgesvd_bufferSize hipsolverDnDgesvd_bufferSize
#define cusolverDnDgesvdj hipsolverDnDgesvdj
#define cusolverDnDgesvdj_bufferSize hipsolverDnDgesvdj_bufferSize
#define cuSolverDnDgesvdj_bufferSize hipSolverDnDgesvdj_bufferSize
#define cusolverDnDgetrf hipsolverDnDgetrf
#define cusolverDnDgetrf_bufferSize hipsolverDnDgetrf_bufferSize
#define cusolverDnDgetrs hipsolverDnDgetrs
#define cusolverDnDorgqr hipsolverDnDorgqr
#define cuSolverDnDorgqr hipSolverDnDorgqr
#define cusolverDnDorgqr_bufferSize hipsolverDnDorgqr_bufferSize
#define cusolverDnDormqr hipsolverDnDormqr
#define cusolverDnDormqr_bufferSize hipsolverDnDormqr_bufferSize
#define cusolverDnDpotrf hipsolverDnDpotrf
#define cuSolverDnDpotrf hipSolverDnDpotrf
#define cusolverDnDpotrf_bufferSize hipsolverDnDpotrf_bufferSize
#define cuSolverDnDpotrf_bufferSize hipSolverDnDpotrf_bufferSize
#define cuSolverDnDpotrf_bufferSize hipSolverDnDpotrf_bufferSize
#define cusolverDnDpotrs hipsolverDnDpotrs
#define cuSolverDnDpotrs hipSolverDnDpotrs
#define cusolverDnDsyevd hipsolverDnDsyevd
#define cusolverDnDsyevd_bufferSize hipsolverDnDsyevd_bufferSize
#define cusolverDnDsyevdx hipsolverDnDsyevdx
#define cusolverDnDsyevdx_bufferSize hipsolverDnDsyevdx_bufferSize
#define cusolverDnDsyevj hipsolverDnDsyevj
#define cusolverDnDsyevj_bufferSize hipsolverDnDsyevj_bufferSize
#define cusolverDngesvd_bufferSize hipsolverDngesvd_bufferSize
#define cusolverDngesvdj_bufferSize hipsolverDngesvdj_bufferSize
#define cusolverDnSetStream hipsolverDnSetStream
#define cuSolverDnSgeqrf hipSolverDnSgeqrf
#define cusolverDnSgeqrf hipsolverDnSgeqrf
#define cuSolverDnSgeqrf hipSolverDnSgeqrf
#define cusolverDnSgeqrf_bufferSize hipsolverDnSgeqrf_bufferSize
#define cuSolverDnSgeqrf_bufferSize hipSolverDnSgeqrf_bufferSize
#define cuSolverDnSgeqrf_bufferSize hipSolverDnSgeqrf_bufferSize
#define cusolverDnSgesvd hipsolverDnSgesvd
#define cusolverDnSgesvd_bufferSize hipsolverDnSgesvd_bufferSize
#define cusolverDnSgesvdj hipsolverDnSgesvdj
#define cuSolverDnSgesvdj hipSolverDnSgesvdj
#define cusolverDnSgesvdj_bufferSize hipsolverDnSgesvdj_bufferSize
#define cusolverDnSgetrf hipsolverDnSgetrf
#define cusolverDnSgetrf_bufferSize hipsolverDnSgetrf_bufferSize
#define cusolverDnSgetrs hipsolverDnSgetrs
#define cusolverDnSorgqr hipsolverDnSorgqr
#define cuSolverDnSorgqr hipSolverDnSorgqr
#define cusolverDnSorgqr_bufferSize hipsolverDnSorgqr_bufferSize
#define cusolverDnSormqr hipsolverDnSormqr
#define cusolverDnSormqr_bufferSize hipsolverDnSormqr_bufferSize
#define cuSolverDnSpotrf hipSolverDnSpotrf
#define cusolverDnSpotrf hipsolverDnSpotrf
#define cuSolverDnSpotrf hipSolverDnSpotrf
#define cusolverDnSpotrf_bufferSize hipsolverDnSpotrf_bufferSize
#define cuSolverDnSpotrf_bufferSize hipSolverDnSpotrf_bufferSize
#define cuSolverDnSpotrf_bufferSize hipSolverDnSpotrf_bufferSize
#define cuSolverDnSpotrs hipSolverDnSpotrs
#define cusolverDnSpotrs hipsolverDnSpotrs
#define cuSolverDnSpotrs hipSolverDnSpotrs
#define cusolverDnSsyevd hipsolverDnSsyevd
#define cusolverDnSsyevd_bufferSize hipsolverDnSsyevd_bufferSize
#define cusolverDnSsyevdx hipsolverDnSsyevdx
#define cusolverDnSsyevdx_bufferSize hipsolverDnSsyevdx_bufferSize
#define cusolverDnSsyevj hipsolverDnSsyevj
#define cusolverDnSsyevj_bufferSize hipsolverDnSsyevj_bufferSize
#define cusolverDnsyevd_bufferSize hipsolverDnsyevd_bufferSize
#define cusolverDnXgesvd hipsolverDnXgesvd
#define cusolverDnXgesvdjSetMaxSweeps hipsolverDnXgesvdjSetMaxSweeps
#define cusolverDnXgesvdjSetTolerance hipsolverDnXgesvdjSetTolerance
#define cusolverDnXsyevjGetSweeps hipsolverDnXsyevjGetSweeps
#define cusolverDnXsyevjSetMaxSweeps hipsolverDnXsyevjSetMaxSweeps
#define cusolverDnXsyevjSetTolerance hipsolverDnXsyevjSetTolerance
#define cusolverSpCreate hipsolverSpCreate
#define cusolverSpDestroy hipsolverSpDestroy
#define cusolverSpSetStream hipsolverSpSetStream
