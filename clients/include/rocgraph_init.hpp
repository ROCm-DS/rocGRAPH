/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef ROCGRAPH_INIT_HPP
#define ROCGRAPH_INIT_HPP

#include "rocgraph_host.hpp"
#include "rocgraph_random.hpp"

#include <fstream>

/* ==================================================================================== */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// for complex number, the real/imag part would be initialized with the same value

// Initialize vector with random values
template <typename T>
void rocgraph_init(T*     A,
                   size_t M,
                   size_t N,
                   size_t lda,
                   size_t stride      = 0,
                   size_t batch_count = 1,
                   T      a           = static_cast<T>(0),
                   T      b           = static_cast<T>(1));

// Initialize vector with random integer values
template <typename T>
void rocgraph_init_exact(T*     A,
                         size_t M,
                         size_t N,
                         size_t lda,
                         size_t stride      = 0,
                         size_t batch_count = 1,
                         int    a           = 1,
                         int    b           = 10);

template <typename T>
void rocgraph_init(host_dense_vector<T>& A,
                   size_t                M,
                   size_t                N,
                   size_t                lda,
                   size_t                stride      = 0,
                   size_t                batch_count = 1,
                   T                     a           = static_cast<T>(0),
                   T                     b           = static_cast<T>(1));

// Initialize vector with random integer values
template <typename T>
void rocgraph_init_exact(host_dense_vector<T>& A,
                         size_t                M,
                         size_t                N,
                         size_t                lda,
                         size_t                stride      = 0,
                         size_t                batch_count = 1,
                         int                   a           = 1,
                         int                   b           = 10);

// Initializes graph index vector with nnz entries ranging from start to end
template <typename I>
void rocgraph_init_index(host_dense_vector<I>& x, size_t nnz, size_t start, size_t end);

// Initialize matrix so adjacent entries have alternating sign.
// In gemm if either A or B are initialized with alernating
// sign the reduction sum will be summing positive
// and negative numbers, so it should not get too large.
// This helps reduce floating point inaccuracies for 16bit
// arithmetic where the exponent has only 5 bits, and the
// mantissa 10 bits.
template <typename T>
void rocgraph_init_alternating_sign(host_dense_vector<T>& A,
                                    size_t                M,
                                    size_t                N,
                                    size_t                lda,
                                    size_t                stride      = 0,
                                    size_t                batch_count = 1);

/* ==================================================================================== */
/*! \brief  Initialize an array with random data, with NaN where appropriate */

template <typename T>
void rocgraph_init_nan(T* A, size_t N);

template <typename T>
void rocgraph_init_nan(host_dense_vector<T>& A,
                       size_t                M,
                       size_t                N,
                       size_t                lda,
                       size_t                stride      = 0,
                       size_t                batch_count = 1);

/* ==================================================================================== */
/*! \brief  Generate a random graph matrix in COO format */
template <typename I, typename T>
void rocgraph_init_coo_matrix(host_dense_vector<I>& row_ind,
                              host_dense_vector<I>& col_ind,
                              host_dense_vector<T>& val,
                              I                     M,
                              I                     N,
                              int64_t               nnz,
                              rocgraph_index_base   base,
                              bool                  full_rank = false,
                              bool                  to_int    = false);

/* ==================================================================================== */
/*! \brief  Generate 2D 9pt laplacian on unit square in CSR format */
template <typename I, typename J, typename T>
void rocgraph_init_csr_laplace2d(host_dense_vector<I>& row_ptr,
                                 host_dense_vector<J>& col_ind,
                                 host_dense_vector<T>& val,
                                 int32_t               dim_x,
                                 int32_t               dim_y,
                                 J&                    M,
                                 J&                    N,
                                 I&                    nnz,
                                 rocgraph_index_base   base);

/* ==================================================================================== */
/*! \brief  Generate 2D 9pt laplacian on unit square in COO format */
template <typename I, typename T>
void rocgraph_init_coo_laplace2d(host_dense_vector<I>& row_ind,
                                 host_dense_vector<I>& col_ind,
                                 host_dense_vector<T>& val,
                                 int32_t               dim_x,
                                 int32_t               dim_y,
                                 I&                    M,
                                 I&                    N,
                                 int64_t&              nnz,
                                 rocgraph_index_base   base);

/* ==================================================================================== */
/*! \brief  Generate 3D 27pt laplacian on unit square in CSR format */
template <typename I, typename J, typename T>
void rocgraph_init_csr_laplace3d(host_dense_vector<I>& row_ptr,
                                 host_dense_vector<J>& col_ind,
                                 host_dense_vector<T>& val,
                                 int32_t               dim_x,
                                 int32_t               dim_y,
                                 int32_t               dim_z,
                                 J&                    M,
                                 J&                    N,
                                 I&                    nnz,
                                 rocgraph_index_base   base);

/* ==================================================================================== */
/*! \brief  Generate 3D 27pt laplacian on unit square in COO format */
template <typename I, typename T>
void rocgraph_init_coo_laplace3d(host_dense_vector<I>& row_ind,
                                 host_dense_vector<I>& col_ind,
                                 host_dense_vector<T>& val,
                                 int32_t               dim_x,
                                 int32_t               dim_y,
                                 int32_t               dim_z,
                                 I&                    M,
                                 I&                    N,
                                 int64_t&              nnz,
                                 rocgraph_index_base   base);

/* ============================================================================================ */
/*! \brief  Read matrix from mtx file in CSR format */
template <typename I, typename J, typename T>
void rocgraph_init_csr_mtx(const char*           filename,
                           host_dense_vector<I>& csr_row_ptr,
                           host_dense_vector<J>& csr_col_ind,
                           host_dense_vector<T>& csr_val,
                           J&                    M,
                           J&                    N,
                           I&                    nnz,
                           rocgraph_index_base   base);

/* ============================================================================================ */
/*! \brief  Read matrix from mtx file in COO format */
template <typename I, typename T>
void rocgraph_init_coo_mtx(const char*           filename,
                           host_dense_vector<I>& coo_row_ind,
                           host_dense_vector<I>& coo_col_ind,
                           host_dense_vector<T>& coo_val,
                           I&                    M,
                           I&                    N,
                           int64_t&              nnz,
                           rocgraph_index_base   base);

/* ============================================================================================ */
/*! \brief  Read matrix from smtx file in CSR format */
template <typename I, typename J, typename T>
void rocgraph_init_csr_smtx(const char*           filename,
                            host_dense_vector<I>& csr_row_ptr,
                            host_dense_vector<J>& csr_col_ind,
                            host_dense_vector<T>& csr_val,
                            J&                    M,
                            J&                    N,
                            I&                    nnz,
                            rocgraph_index_base   base);

/* ============================================================================================ */
/*! \brief  Read matrix from smtx file in COO format */
template <typename I, typename T>
void rocgraph_init_coo_smtx(const char*           filename,
                            host_dense_vector<I>& coo_row_ind,
                            host_dense_vector<I>& coo_col_ind,
                            host_dense_vector<T>& coo_val,
                            I&                    M,
                            I&                    N,
                            int64_t&              nnz,
                            rocgraph_index_base   base);

/* ==================================================================================== */
/*! \brief  Read matrix from binary file in rocALUTION format */
template <typename I, typename J, typename T>
void rocgraph_init_csr_rocalution(const char*           filename,
                                  host_dense_vector<I>& row_ptr,
                                  host_dense_vector<J>& col_ind,
                                  host_dense_vector<T>& val,
                                  J&                    M,
                                  J&                    N,
                                  I&                    nnz,
                                  rocgraph_index_base   base);

/* ==================================================================================== */
/*! \brief  Read matrix from binary file in rocALUTION format */
template <typename I, typename T>
void rocgraph_init_coo_rocalution(const char*           filename,
                                  host_dense_vector<I>& row_ind,
                                  host_dense_vector<I>& col_ind,
                                  host_dense_vector<T>& val,
                                  I&                    M,
                                  I&                    N,
                                  int64_t&              nnz,
                                  rocgraph_index_base   base);

/* ==================================================================================== */
/*! \brief  Generate a random graph matrix in CSR format */
template <typename I, typename J, typename T>
void rocgraph_init_csr_random(host_dense_vector<I>&     row_ptr,
                              host_dense_vector<J>&     col_ind,
                              host_dense_vector<T>&     val,
                              J                         M,
                              J                         N,
                              I&                        nnz,
                              rocgraph_index_base       base,
                              rocgraph_matrix_init_kind init_kind,
                              bool                      full_rank = false,
                              bool                      to_int    = false);

/* ==================================================================================== */
/*! \brief  Generate a random graph matrix in COO format */
template <typename I, typename T>
void rocgraph_init_coo_random(host_dense_vector<I>&     row_ind,
                              host_dense_vector<I>&     col_ind,
                              host_dense_vector<T>&     val,
                              I                         M,
                              I                         N,
                              int64_t&                  nnz,
                              rocgraph_index_base       base,
                              rocgraph_matrix_init_kind init_kind,
                              bool                      full_rank = false,
                              bool                      to_int    = false);

/* ==================================================================================== */
/*! \brief  Generate a tridiagonal graph matrix in COO format */
template <typename I, typename T>
void rocgraph_init_coo_tridiagonal(host_dense_vector<I>& row_ind,
                                   host_dense_vector<I>& col_ind,
                                   host_dense_vector<T>& val,
                                   I                     M,
                                   I                     N,
                                   int64_t&              nnz,
                                   rocgraph_index_base   base,
                                   I                     l,
                                   I                     u);

/* ==================================================================================== */
/*! \brief  Generate a tridiagonal graph matrix in CSR format */
template <typename I, typename J, typename T>
void rocgraph_init_csr_tridiagonal(host_dense_vector<I>& row_ptr,
                                   host_dense_vector<J>& col_ind,
                                   host_dense_vector<T>& val,
                                   J                     M,
                                   J                     N,
                                   I&                    nnz,
                                   rocgraph_index_base   base,
                                   J                     l,
                                   J                     u);

/* ==================================================================================== */
/*! \brief  Generate a penta diagonal graph matrix in COO format */
template <typename I, typename T>
void rocgraph_init_coo_pentadiagonal(host_dense_vector<I>& row_ind,
                                     host_dense_vector<I>& col_ind,
                                     host_dense_vector<T>& val,
                                     I                     M,
                                     I                     N,
                                     int64_t&              nnz,
                                     rocgraph_index_base   base,
                                     I                     ll,
                                     I                     l,
                                     I                     u,
                                     I                     uu);

/* ==================================================================================== */
/*! \brief  Generate a penta diagonal graph matrix in CSR format */
template <typename I, typename J, typename T>
void rocgraph_init_csr_pentadiagonal(host_dense_vector<I>& row_ptr,
                                     host_dense_vector<J>& col_ind,
                                     host_dense_vector<T>& val,
                                     J                     M,
                                     J                     N,
                                     I&                    nnz,
                                     rocgraph_index_base   base,
                                     J                     ll,
                                     J                     l,
                                     J                     u,
                                     J                     uu);

#endif // ROCGRAPH_INIT_HPP
