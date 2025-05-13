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

#include <hipblaslt/hipblaslt.h>

// types
#define cublasLtHandle_t hipblasLtHandle_t
#define cublasLtMatrixLayout_t hipblasLtMatrixLayout_t
#define cublasLtMatmulDesc_t hipblasLtMatmulDesc_t
#define cublasLtMatmulHeuristicResult_t hipblasLtMatmulHeuristicResult_t
#define cublasLtMatmulPreference_t hipblasLtMatmulPreference_t

#define cudaDataType hipDataType

// macros
#define CUBLASLT_MATMUL_DESC_POINTER_MODE HIPBLASLT_MATMUL_DESC_POINTER_MODE
#define CUBLASLT_MATMUL_DESC_TRANSA HIPBLASLT_MATMUL_DESC_TRANSA
#define CUBLASLT_MATMUL_DESC_TRANSB HIPBLASLT_MATMUL_DESC_TRANSB

// functions
#define cublasLtCreate hipblasLtCreate
#define cublasLtDestroy hipblasLtDestroy
#define cublasLtMatrixLayoutCreate hipblasLtMatrixLayoutCreate
#define cublasLtMatrixLayoutDestroy hipblasLtMatrixLayoutDestroy
#define cublasLtMatmulDescCreate hipblasLtMatmulDescCreate
#define cublasLtMatmulDescDestroy hipblasLtMatmulDescDestroy
#define cublasLtMatmulPreferenceCreate hipblasLtMatmulPreferenceCreate
#define cublasLtMatmulAlgoGetHeuristic hipblasLtMatmulAlgoGetHeuristic
#define cublasLtMatmulPreferenceDestroy hipblasLtMatmulPreferenceDestroy
#define cublasLtMatmulDescSetAttribute hipblasLtMatmulDescSetAttribute
#define cublasLtMatmul hipblasLtMatmul
