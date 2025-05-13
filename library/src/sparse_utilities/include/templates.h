/*! \file */

/*
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "sparse_utility_types.h"

#include "internal/level2/sparse_utility_bsrmv.h"
#include "internal/level2/sparse_utility_csrsv.h"
#include "internal/level2/sparse_utility_gebsrmv.h"
#include "internal/level3/sparse_utility_csrmm.h"

//
// For reusing without recompiling.
// e.g. call rocgraph_bsrmv rather than rocgraph_bsrmv_template.
//

//
// csrsv_buffer_size.
//
template <typename T>
inline rocgraph_status rocgraph_csrsv_buffer_size(rocgraph_handle          handle,
                                                  rocgraph_operation       trans,
                                                  rocgraph_int             m,
                                                  rocgraph_int             nnz,
                                                  const rocgraph_mat_descr descr,
                                                  const T*                 csr_val,
                                                  const rocgraph_int*      csr_row_ptr,
                                                  const rocgraph_int*      csr_col_ind,
                                                  rocgraph_mat_info        info,
                                                  size_t*                  buffer_size);

#define SPZL(NAME, TYPE)                                                                         \
    template <>                                                                                  \
    inline rocgraph_status rocgraph_csrsv_buffer_size(rocgraph_handle          handle,           \
                                                      rocgraph_operation       trans,            \
                                                      rocgraph_int             m,                \
                                                      rocgraph_int             nnz,              \
                                                      const rocgraph_mat_descr descr,            \
                                                      const TYPE*              csr_val,          \
                                                      const rocgraph_int*      csr_row_ptr,      \
                                                      const rocgraph_int*      csr_col_ind,      \
                                                      rocgraph_mat_info        info,             \
                                                      size_t*                  buffer_size)      \
    {                                                                                            \
        return NAME(                                                                             \
            handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size); \
    }

SPZL(rocgraph_scsrsv_buffer_size, float);
SPZL(rocgraph_dcsrsv_buffer_size, double);
#undef SPZL

// bsrmv
template <typename T>
inline rocgraph_status rocgraph_bsrmv(rocgraph_handle          handle,
                                      rocgraph_direction       dir,
                                      rocgraph_operation       trans,
                                      rocgraph_int             mb,
                                      rocgraph_int             nb,
                                      rocgraph_int             nnzb,
                                      const T*                 alpha,
                                      const rocgraph_mat_descr descr,
                                      const T*                 bsr_val,
                                      const rocgraph_int*      bsr_row_ptr,
                                      const rocgraph_int*      bsr_col_ind,
                                      rocgraph_int             bsr_dim,
                                      rocgraph_mat_info        info,
                                      const T*                 x,
                                      const T*                 beta,
                                      T*                       y);

template <>
inline rocgraph_status rocgraph_bsrmv(rocgraph_handle          handle,
                                      rocgraph_direction       dir,
                                      rocgraph_operation       trans,
                                      rocgraph_int             mb,
                                      rocgraph_int             nb,
                                      rocgraph_int             nnzb,
                                      const float*             alpha,
                                      const rocgraph_mat_descr descr,
                                      const float*             bsr_val,
                                      const rocgraph_int*      bsr_row_ptr,
                                      const rocgraph_int*      bsr_col_ind,
                                      rocgraph_int             bsr_dim,
                                      rocgraph_mat_info        info,
                                      const float*             x,
                                      const float*             beta,
                                      float*                   y)
{
    return rocgraph_sbsrmv(handle,
                           dir,
                           trans,
                           mb,
                           nb,
                           nnzb,
                           alpha,
                           descr,
                           bsr_val,
                           bsr_row_ptr,
                           bsr_col_ind,
                           bsr_dim,
                           info,
                           x,
                           beta,
                           y);
}

template <>
inline rocgraph_status rocgraph_bsrmv(rocgraph_handle          handle,
                                      rocgraph_direction       dir,
                                      rocgraph_operation       trans,
                                      rocgraph_int             mb,
                                      rocgraph_int             nb,
                                      rocgraph_int             nnzb,
                                      const double*            alpha,
                                      const rocgraph_mat_descr descr,
                                      const double*            bsr_val,
                                      const rocgraph_int*      bsr_row_ptr,
                                      const rocgraph_int*      bsr_col_ind,
                                      rocgraph_int             bsr_dim,
                                      rocgraph_mat_info        info,
                                      const double*            x,
                                      const double*            beta,
                                      double*                  y)
{
    return rocgraph_dbsrmv(handle,
                           dir,
                           trans,
                           mb,
                           nb,
                           nnzb,
                           alpha,
                           descr,
                           bsr_val,
                           bsr_row_ptr,
                           bsr_col_ind,
                           bsr_dim,
                           info,
                           x,
                           beta,
                           y);
}

// gebsrmv
template <typename T>
inline rocgraph_status rocgraph_gebsrmv(rocgraph_handle          handle,
                                        rocgraph_direction       dir,
                                        rocgraph_operation       trans,
                                        rocgraph_int             mb,
                                        rocgraph_int             nb,
                                        rocgraph_int             nnzb,
                                        const T*                 alpha,
                                        const rocgraph_mat_descr descr,
                                        const T*                 bsr_val,
                                        const rocgraph_int*      bsr_row_ptr,
                                        const rocgraph_int*      bsr_col_ind,
                                        rocgraph_int             row_block_dim,
                                        rocgraph_int             col_block_dim,
                                        const T*                 x,
                                        const T*                 beta,
                                        T*                       y);

template <>
inline rocgraph_status rocgraph_gebsrmv(rocgraph_handle          handle,
                                        rocgraph_direction       dir,
                                        rocgraph_operation       trans,
                                        rocgraph_int             mb,
                                        rocgraph_int             nb,
                                        rocgraph_int             nnzb,
                                        const float*             alpha,
                                        const rocgraph_mat_descr descr,
                                        const float*             bsr_val,
                                        const rocgraph_int*      bsr_row_ptr,
                                        const rocgraph_int*      bsr_col_ind,
                                        rocgraph_int             row_block_dim,
                                        rocgraph_int             col_block_dim,
                                        const float*             x,
                                        const float*             beta,
                                        float*                   y)
{
    return rocgraph_sgebsrmv(handle,
                             dir,
                             trans,
                             mb,
                             nb,
                             nnzb,
                             alpha,
                             descr,
                             bsr_val,
                             bsr_row_ptr,
                             bsr_col_ind,
                             row_block_dim,
                             col_block_dim,
                             x,
                             beta,
                             y);
}

template <>
inline rocgraph_status rocgraph_gebsrmv(rocgraph_handle          handle,
                                        rocgraph_direction       dir,
                                        rocgraph_operation       trans,
                                        rocgraph_int             mb,
                                        rocgraph_int             nb,
                                        rocgraph_int             nnzb,
                                        const double*            alpha,
                                        const rocgraph_mat_descr descr,
                                        const double*            bsr_val,
                                        const rocgraph_int*      bsr_row_ptr,
                                        const rocgraph_int*      bsr_col_ind,
                                        rocgraph_int             row_block_dim,
                                        rocgraph_int             col_block_dim,
                                        const double*            x,
                                        const double*            beta,
                                        double*                  y)
{
    return rocgraph_dgebsrmv(handle,
                             dir,
                             trans,
                             mb,
                             nb,
                             nnzb,
                             alpha,
                             descr,
                             bsr_val,
                             bsr_row_ptr,
                             bsr_col_ind,
                             row_block_dim,
                             col_block_dim,
                             x,
                             beta,
                             y);
}

// csrmm
template <typename T>
inline rocgraph_status rocgraph_csrmm(rocgraph_handle          handle,
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
                                      int64_t                  ldb,
                                      const T*                 beta,
                                      T*                       C,
                                      int64_t                  ldc);

template <>
inline rocgraph_status rocgraph_csrmm(rocgraph_handle          handle,
                                      rocgraph_operation       trans_A,
                                      rocgraph_operation       trans_B,
                                      rocgraph_int             m,
                                      rocgraph_int             n,
                                      rocgraph_int             k,
                                      rocgraph_int             nnz,
                                      const float*             alpha,
                                      const rocgraph_mat_descr descr,
                                      const float*             csr_val,
                                      const rocgraph_int*      csr_row_ptr,
                                      const rocgraph_int*      csr_col_ind,
                                      const float*             B,
                                      int64_t                  ldb,
                                      const float*             beta,
                                      float*                   C,
                                      int64_t                  ldc)
{
    return rocgraph_scsrmm(handle,
                           trans_A,
                           trans_B,
                           m,
                           n,
                           k,
                           nnz,
                           alpha,
                           descr,
                           csr_val,
                           csr_row_ptr,
                           csr_col_ind,
                           B,
                           ldb,
                           beta,
                           C,
                           ldc);
}

template <>
inline rocgraph_status rocgraph_csrmm(rocgraph_handle          handle,
                                      rocgraph_operation       trans_A,
                                      rocgraph_operation       trans_B,
                                      rocgraph_int             m,
                                      rocgraph_int             n,
                                      rocgraph_int             k,
                                      rocgraph_int             nnz,
                                      const double*            alpha,
                                      const rocgraph_mat_descr descr,
                                      const double*            csr_val,
                                      const rocgraph_int*      csr_row_ptr,
                                      const rocgraph_int*      csr_col_ind,
                                      const double*            B,
                                      int64_t                  ldb,
                                      const double*            beta,
                                      double*                  C,
                                      int64_t                  ldc)
{
    return rocgraph_dcsrmm(handle,
                           trans_A,
                           trans_B,
                           m,
                           n,
                           k,
                           nnz,
                           alpha,
                           descr,
                           csr_val,
                           csr_row_ptr,
                           csr_col_ind,
                           B,
                           ldb,
                           beta,
                           C,
                           ldc);
}
