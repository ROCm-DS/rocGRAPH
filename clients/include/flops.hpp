/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/*! \file
 *  \brief flops.hpp provides floating point counts of Graph Linear Algebra Subprograms
 *  of Level 1, 2 and 3.
 */

#pragma once
#ifndef FLOPS_HPP
#define FLOPS_HPP

#include <rocgraph.h>

template <typename I, typename J>
constexpr double spmv_gflop_count(J M, I nnz, bool beta = false)
{
    return (2.0 * nnz + (beta ? M : 0)) / 1e9;
}

template <typename I, typename J>
constexpr double csrmm_gflop_count(J N, I nnz_A, I nnz_C, bool beta = false)
{
    // Multiplication by 2 comes from 1 addition and 1 multiplication in product. Multiplication
    // by alpha and beta not counted.
    return (2.0 * nnz_A * N + (beta ? nnz_C : 0)) / 1e9;
}

template <typename I, typename J>
constexpr double spmm_gflop_count(J N, I nnz_A, I nnz_C, bool beta = false)
{
    return csrmm_gflop_count(N, nnz_A, nnz_C, beta);
}

#endif // FLOPS_HPP
