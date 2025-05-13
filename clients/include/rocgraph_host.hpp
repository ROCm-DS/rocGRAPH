/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef ROCGRAPH_HOST_HPP
#define ROCGRAPH_HOST_HPP

#include "rocgraph_test.hpp"
#include "rocgraph_vector.hpp"

#include <hip/hip_runtime_api.h>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T, typename I, typename J>
struct rocgraph_host
{

    static void csrddmm(rocgraph_operation  trans_A,
                        rocgraph_operation  trans_B,
                        rocgraph_order      order_A,
                        rocgraph_order      order_B,
                        J                   M,
                        J                   N,
                        J                   K,
                        I                   nnz,
                        const T*            alpha,
                        const T*            A,
                        int64_t             lda,
                        const T*            B,
                        int64_t             ldb,
                        const T*            beta,
                        const I*            csr_row_ptr_C,
                        const J*            csr_col_ind_C,
                        T*                  csr_val_C,
                        rocgraph_index_base base_C);

    static void cscddmm(rocgraph_operation  trans_A,
                        rocgraph_operation  trans_B,
                        rocgraph_order      order_A,
                        rocgraph_order      order_B,
                        J                   M,
                        J                   N,
                        J                   K,
                        I                   nnz,
                        const T*            alpha,
                        const T*            A,
                        int64_t             lda,
                        const T*            B,
                        int64_t             ldb,
                        const T*            beta,
                        const I*            csr_row_ptr_C,
                        const J*            csr_col_ind_C,
                        T*                  csr_val_C,
                        rocgraph_index_base base_C);

    static void cooddmm(rocgraph_operation  trans_A,
                        rocgraph_operation  trans_B,
                        rocgraph_order      order_A,
                        rocgraph_order      order_B,
                        J                   M,
                        J                   N,
                        J                   K,
                        I                   nnz,
                        const T*            alpha,
                        const T*            A,
                        int64_t             lda,
                        const T*            B,
                        int64_t             ldb,
                        const T*            beta,
                        const I*            coo_row_ind_C,
                        const I*            coo_col_ind_C,
                        T*                  coo_val_C,
                        rocgraph_index_base base_C);

    static void cooaosddmm(rocgraph_operation  trans_A,
                           rocgraph_operation  trans_B,
                           rocgraph_order      order_A,
                           rocgraph_order      order_B,
                           J                   M,
                           J                   N,
                           J                   K,
                           I                   nnz,
                           const T*            alpha,
                           const T*            A,
                           int64_t             lda,
                           const T*            B,
                           int64_t             ldb,
                           const T*            beta,
                           const I*            coo_row_ind_C,
                           const I*            coo_col_ind_C,
                           T*                  coo_val_C,
                           rocgraph_index_base base_C);

    static void ellddmm(rocgraph_operation  trans_A,
                        rocgraph_operation  trans_B,
                        rocgraph_order      order_A,
                        rocgraph_order      order_B,
                        J                   M,
                        J                   N,
                        J                   K,
                        I                   nnz,
                        const T*            alpha,
                        const T*            A,
                        int64_t             lda,
                        const T*            B,
                        int64_t             ldb,
                        const T*            beta,
                        const J             ell_width,
                        const I*            ell_ind_C,
                        T*                  ell_val_C,
                        rocgraph_index_base base_C);
};

template <typename T, typename I, typename A, typename X, typename Y>
void host_coomv(rocgraph_operation  trans,
                I                   M,
                I                   N,
                int64_t             nnz,
                T                   alpha,
                const I*            coo_row_ind,
                const I*            coo_col_ind,
                const A*            coo_val,
                const X*            x,
                T                   beta,
                Y*                  y,
                rocgraph_index_base base);

template <typename T, typename I, typename J, typename A, typename X, typename Y>
void host_csrmv(rocgraph_operation   trans,
                J                    M,
                J                    N,
                I                    nnz,
                T                    alpha,
                const I*             csr_row_ptr,
                const J*             csr_col_ind,
                const A*             csr_val,
                const X*             x,
                T                    beta,
                Y*                   y,
                rocgraph_index_base  base,
                rocgraph_matrix_type matrix_type,
                rocgraph_spmv_alg    algo,
                bool                 force_conj);

template <typename T, typename I, typename J, typename A, typename X, typename Y>
void host_cscmv(rocgraph_operation trans,
                J                  M,
                J                  N,
                I                  nnz,
                T                  alpha,
                const I* __restrict csc_col_ptr,
                const J* __restrict csc_row_ind,
                const A* __restrict csc_val,
                const X* __restrict x,
                T beta,
                Y* __restrict y,
                rocgraph_index_base  base,
                rocgraph_matrix_type matrix_type,
                rocgraph_spmv_alg    algo);

template <typename T, typename I = rocgraph_int, typename J = rocgraph_int>
void host_csrmm(J                   M,
                J                   N,
                J                   K,
                rocgraph_operation  transA,
                rocgraph_operation  transB,
                T                   alpha,
                const I*            csr_row_ptr_A,
                const J*            csr_col_ind_A,
                const T*            csr_val_A,
                const T*            B,
                int64_t             ldb,
                rocgraph_order      order_B,
                T                   beta,
                T*                  C,
                int64_t             ldc,
                rocgraph_order      order_C,
                rocgraph_index_base base,
                bool                force_conj_A);

template <typename T, typename I>
void host_coomm(I                   M,
                I                   N,
                I                   K,
                int64_t             nnz,
                rocgraph_operation  transA,
                rocgraph_operation  transB,
                T                   alpha,
                const I*            coo_row_ind_A,
                const I*            coo_col_ind_A,
                const T*            coo_val_A,
                const T*            B,
                int64_t             ldb,
                rocgraph_order      order_B,
                T                   beta,
                T*                  C,
                int64_t             ldc,
                rocgraph_order      order_C,
                rocgraph_index_base base);

template <typename T, typename I = rocgraph_int, typename J = rocgraph_int>
void host_cscmm(J                  M,
                J                  N,
                J                  K,
                rocgraph_operation transA,
                rocgraph_operation transB,
                T                  alpha,
                const I* __restrict csc_col_ptr_A,
                const J* __restrict csc_row_ind_A,
                const T* __restrict csc_val_A,
                const T* __restrict B,
                int64_t        ldb,
                rocgraph_order order_B,
                T              beta,
                T* __restrict C,
                int64_t             ldc,
                rocgraph_order      order_C,
                rocgraph_index_base base);

/*
 * ===========================================================================
 *    conversion GRAPH
 * ===========================================================================
 */
template <typename T>
rocgraph_status host_nnz(rocgraph_direction dirA,
                         rocgraph_int       m,
                         rocgraph_int       n,
                         const T*           A,
                         int64_t            lda,
                         rocgraph_int*      nnz_per_row_columns,
                         rocgraph_int*      nnz_total_dev_host_ptr);

template <typename I, typename J>
void host_csr_to_coo(J                           M,
                     I                           nnz,
                     const host_dense_vector<I>& csr_row_ptr,
                     host_dense_vector<J>&       coo_row_ind,
                     rocgraph_index_base         base);

template <typename I, typename J>
void host_csr_to_coo_aos(J                           M,
                         I                           nnz,
                         const host_dense_vector<I>& csr_row_ptr,
                         const host_dense_vector<J>& csr_col_ind,
                         host_dense_vector<I>&       coo_ind,
                         rocgraph_index_base         base);

template <typename I, typename J, typename T>
void host_csr_to_csc(J                     M,
                     J                     N,
                     I                     nnz,
                     const I*              csr_row_ptr,
                     const J*              csr_col_ind,
                     const T*              csr_val,
                     host_dense_vector<J>& csc_row_ind,
                     host_dense_vector<I>& csc_col_ptr,
                     host_dense_vector<T>& csc_val,
                     rocgraph_action       action,
                     rocgraph_index_base   base);

template <typename I, typename J>
void host_coo_to_csr(J M, I nnz, const J* coo_row_ind, I* csr_row_ptr, rocgraph_index_base base);

template <typename T>
void host_coosort_by_column(rocgraph_int                     M,
                            rocgraph_int                     nnz,
                            host_dense_vector<rocgraph_int>& coo_row_ind,
                            host_dense_vector<rocgraph_int>& coo_col_ind,
                            host_dense_vector<T>&            coo_val);

#endif // ROCGRAPH_HOST_HPP
