/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/*! \file
 *  \brief rocgraph.hpp exposes C++ templated Graph Linear Algebra interface
 *  with only the precision templated.
 */

#pragma once
#ifndef ROCGRAPH_HPP
#define ROCGRAPH_HPP

#include "rocgraph_sparse_utility.hpp"
#include "rocgraph_traits.hpp"
#include <rocgraph.h>

#define REAL_TEMPLATE(NAME_, ...)                            \
    template <typename T>                                    \
    rocgraph_status (*rocgraph_##NAME_)(__VA_ARGS__);        \
    template <>                                              \
    static auto rocgraph_##NAME_<float> = rocgraph_s##NAME_; \
    template <>                                              \
    static auto rocgraph_##NAME_<double> = rocgraph_d##NAME_

#define COMPLEX_TEMPLATE(NAME_, ...) (void)0

#define REAL_COMPLEX_TEMPLATE(NAME_, ...)                    \
    template <typename T>                                    \
    rocgraph_status (*rocgraph_##NAME_)(__VA_ARGS__);        \
    template <>                                              \
    static auto rocgraph_##NAME_<float> = rocgraph_s##NAME_; \
    template <>                                              \
    static auto rocgraph_##NAME_<double> = rocgraph_d##NAME_

/*
 * ===========================================================================
 *    utility GRAPH
 * ===========================================================================
 */

// gthr
REAL_COMPLEX_TEMPLATE(gthr,
                      rocgraph_handle     handle,
                      rocgraph_int        nnz,
                      const T*            y,
                      T*                  x_val,
                      const rocgraph_int* x_ind,
                      rocgraph_index_base idx_base);

/*
 * ===========================================================================
 *    level 2 GRAPH
 * ===========================================================================
 */

// coomv
REAL_COMPLEX_TEMPLATE(coomv,
                      rocgraph_handle          handle,
                      rocgraph_operation       trans,
                      rocgraph_int             m,
                      rocgraph_int             n,
                      rocgraph_int             nnz,
                      const T*                 alpha,
                      const rocgraph_mat_descr descr,
                      const T*                 coo_val,
                      const rocgraph_int*      coo_row_ind,
                      const rocgraph_int*      coo_col_ind,
                      const T*                 x,
                      const T*                 beta,
                      T*                       y);

// csrmv
REAL_COMPLEX_TEMPLATE(csrmv_analysis,
                      rocgraph_handle          handle,
                      rocgraph_operation       trans,
                      rocgraph_int             m,
                      rocgraph_int             n,
                      rocgraph_int             nnz,
                      const rocgraph_mat_descr descr,
                      const T*                 csr_val,
                      const rocgraph_int*      csr_row_ptr,
                      const rocgraph_int*      csr_col_ind,
                      rocgraph_mat_info        info);

REAL_COMPLEX_TEMPLATE(csrmv,
                      rocgraph_handle          handle,
                      rocgraph_operation       trans,
                      rocgraph_int             m,
                      rocgraph_int             n,
                      rocgraph_int             nnz,
                      const T*                 alpha,
                      const rocgraph_mat_descr descr,
                      const T*                 csr_val,
                      const rocgraph_int*      csr_row_ptr,
                      const rocgraph_int*      csr_col_ind,
                      rocgraph_mat_info        info,
                      const T*                 x,
                      const T*                 beta,
                      T*                       y);

/*
 * ===========================================================================
 *    level 3 GRAPH
 * ===========================================================================
 */

// csrmm
REAL_COMPLEX_TEMPLATE(csrmm,
                      rocgraph_handle          handle,
                      rocgraph_operation       trans_A,
                      rocgraph_operation       trans_B,
                      rocgraph_int             m,
                      rocgraph_int             n,
                      rocgraph_int             k,
                      rocgraph_int             nnz,
                      const T*                 alpha,
                      const rocgraph_mat_descr descr,
                      const T*                 csr_val,
                      const rocgraph_int*      csr_row_ptr,
                      const rocgraph_int*      csr_col_ind,
                      const T*                 B,
                      rocgraph_int             ldb,
                      const T*                 beta,
                      T*                       C,
                      rocgraph_int             ldc);

/*
 * ===========================================================================
 *    conversion GRAPH
 * ===========================================================================
 */

// nnz
REAL_COMPLEX_TEMPLATE(nnz,
                      rocgraph_handle          handle,
                      rocgraph_direction       dir,
                      rocgraph_int             m,
                      rocgraph_int             n,
                      const rocgraph_mat_descr descr,
                      const T*                 A,
                      rocgraph_int             lda,
                      rocgraph_int*            nnz_per_row_columns,
                      rocgraph_int*            nnz_total_dev_host_ptr);

// nnz_compress
REAL_COMPLEX_TEMPLATE(nnz_compress,
                      rocgraph_handle          handle,
                      rocgraph_int             m,
                      const rocgraph_mat_descr descr_A,
                      const T*                 csr_val_A,
                      const rocgraph_int*      csr_row_ptr_A,
                      rocgraph_int*            nnz_per_row,
                      rocgraph_int*            nnz_C,
                      T                        tol);

// csr2csc
REAL_COMPLEX_TEMPLATE(csr2csc,
                      rocgraph_handle     handle,
                      rocgraph_int        m,
                      rocgraph_int        n,
                      rocgraph_int        nnz,
                      const T*            csr_val,
                      const rocgraph_int* csr_row_ptr,
                      const rocgraph_int* csr_col_ind,
                      T*                  csc_val,
                      rocgraph_int*       csc_row_ind,
                      rocgraph_int*       csc_col_ptr,
                      rocgraph_action     copy_values,
                      rocgraph_index_base idx_base,
                      void*               temp_buffer);

/*
 * ===========================================================================
 *    reordering GRAPH
 * ===========================================================================
 */

// csrcolor
REAL_COMPLEX_TEMPLATE(csrcolor,
                      rocgraph_handle          handle,
                      rocgraph_int             m,
                      rocgraph_int             nnz,
                      const rocgraph_mat_descr descr,
                      const T*                 csr_val,
                      const rocgraph_int*      csr_row_ptr,
                      const rocgraph_int*      csr_col_ind,
                      const T*                 fraction_to_color,
                      rocgraph_int*            ncolors,
                      rocgraph_int*            coloring,
                      rocgraph_int*            reordering,
                      rocgraph_mat_info        info);

#endif // ROCGRAPH_HPP
