/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/*! \file
 *  \brief gbyte.hpp provides data transfer counts of Graph Linear Algebra Subprograms
 *  of Level 1, 2 and 3.
 */

#pragma once
#ifndef GBYTE_HPP
#define GBYTE_HPP

#include <rocgraph.h>

template <typename A, typename X, typename Y, typename I>
constexpr double coomv_gbyte_count(I M, I N, int64_t nnz, bool beta = false)
{
    return (sizeof(I) * 2.0 * nnz + sizeof(A) * nnz + sizeof(Y) * (M + (beta ? M : 0))
            + sizeof(X) * N)
           / 1e9;
}

template <typename T, typename I>
constexpr double coomv_gbyte_count(I M, I N, int64_t nnz, bool beta = false)
{
    return coomv_gbyte_count<T, T, T>(M, N, nnz, beta);
}

template <typename A, typename X, typename Y, typename I, typename J>
constexpr double csrmv_gbyte_count(J M, J N, I nnz, bool beta = false)
{
    return (sizeof(I) * (M + 1) + sizeof(J) * nnz + sizeof(A) * nnz
            + sizeof(Y) * (M + (beta ? M : 0)) + sizeof(X) * N)
           / 1e9;
}

template <typename T, typename I, typename J>
constexpr double csrmv_gbyte_count(J M, J N, I nnz, bool beta = false)
{
    return csrmv_gbyte_count<T, T, T>(M, N, nnz, beta);
}

template <typename A, typename X, typename Y, typename I, typename J>
constexpr double cscmv_gbyte_count(J M, J N, I nnz, bool beta = false)
{
    return (sizeof(I) * (N + 1) + sizeof(J) * nnz + sizeof(A) * nnz
            + sizeof(Y) * (M + (beta ? M : 0)) + sizeof(X) * N)
           / 1e9;
}

template <typename T, typename I, typename J>
constexpr double cscmv_gbyte_count(J M, J N, I nnz, bool beta = false)
{
    return cscmv_gbyte_count<T, T, T>(M, N, nnz, beta);
}

template <typename T, typename I, typename J>
constexpr double csrmm_gbyte_count(J M, I nnz_A, I nnz_B, I nnz_C, bool beta = false)
{
    return ((M + 1) * sizeof(I) + nnz_A * sizeof(J)
            + (nnz_A + nnz_B + nnz_C + (beta ? nnz_C : 0)) * sizeof(T))
           / 1e9;
}

template <typename T, typename I, typename J>
constexpr double cscmm_gbyte_count(J N, I nnz_A, I nnz_B, I nnz_C, bool beta = false)
{
    return csrmm_gbyte_count<T>(N, nnz_A, nnz_B, nnz_C, beta);
}

template <typename T, typename I>
constexpr double coomm_gbyte_count(int64_t nnz_A, int64_t nnz_B, int64_t nnz_C, bool beta = false)
{
    return (2.0 * nnz_A * sizeof(I) + (nnz_A + nnz_B + nnz_C + (beta ? nnz_C : 0)) * sizeof(T))
           / 1e9;
}
/*
 * ===========================================================================
 *    conversion GRAPH
 * ===========================================================================
 */
template <typename T>
constexpr double nnz_gbyte_count(rocgraph_int M, rocgraph_int N, rocgraph_direction dir)
{
    return ((M * N) * sizeof(T) + ((rocgraph_direction_row == dir) ? M : N) * sizeof(rocgraph_int))
           / 1e9;
}

template <typename T>
constexpr double csr2coo_gbyte_count(rocgraph_int M, rocgraph_int nnz)
{
    return (M + 1 + nnz) * sizeof(rocgraph_int) / 1e9;
}

template <typename T>
constexpr double coo2csr_gbyte_count(rocgraph_int M, rocgraph_int nnz)
{
    return (M + 1 + nnz) * sizeof(rocgraph_int) / 1e9;
}

template <typename T>
constexpr double
    csr2csc_gbyte_count(rocgraph_int M, rocgraph_int N, rocgraph_int nnz, rocgraph_action action)
{
    return ((M + N + 2 + 2.0 * nnz) * sizeof(rocgraph_int)
            + (action == rocgraph_action_numeric ? (2.0 * nnz) * sizeof(T) : 0.0))
           / 1e9;
}

template <typename T>
constexpr double csrsort_gbyte_count(rocgraph_int M, rocgraph_int nnz, bool permute)
{
    return ((2.0 * M + 2.0 + 2.0 * nnz + (permute ? 2.0 * nnz : 0.0)) * sizeof(rocgraph_int)) / 1e9;
}

template <typename T>
constexpr double cscsort_gbyte_count(rocgraph_int N, rocgraph_int nnz, bool permute)
{
    return ((2.0 * N + 2.0 + 2.0 * nnz + (permute ? 2.0 * nnz : 0.0)) * sizeof(rocgraph_int)) / 1e9;
}

template <typename T>
constexpr double coosort_gbyte_count(rocgraph_int nnz, bool permute)
{
    return ((4.0 * nnz + (permute ? 2.0 * nnz : 0.0)) * sizeof(rocgraph_int)) / 1e9;
}

#endif // GBYTE_HPP
