/*! \file */

// Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_init.hpp"

#include "rocgraph_matrix_factory_zero.hpp"

template <typename T, typename I, typename J>
rocgraph_matrix_factory_zero<T, I, J>::rocgraph_matrix_factory_zero(){};

template <typename T, typename I, typename J>
void rocgraph_matrix_factory_zero<T, I, J>::init_csr(host_dense_vector<I>& csr_row_ptr,
                                                     host_dense_vector<J>& csr_col_ind,
                                                     host_dense_vector<T>& csr_val,
                                                     J&                    M,
                                                     J&                    N,
                                                     I&                    nnz,
                                                     rocgraph_index_base   base,
                                                     rocgraph_matrix_type  matrix_type,
                                                     rocgraph_fill_mode    uplo,
                                                     rocgraph_storage_mode storage)
{
    csr_row_ptr.resize((M > 0) ? (M + 1) : 0, static_cast<I>(base));
    csr_col_ind.resize(0);
    csr_val.resize(0);

    nnz = 0;
}

template <typename T, typename I, typename J>
void rocgraph_matrix_factory_zero<T, I, J>::init_coo(host_dense_vector<I>& coo_row_ind,
                                                     host_dense_vector<I>& coo_col_ind,
                                                     host_dense_vector<T>& coo_val,
                                                     I&                    M,
                                                     I&                    N,
                                                     int64_t&              nnz,
                                                     rocgraph_index_base   base,
                                                     rocgraph_matrix_type  matrix_type,
                                                     rocgraph_fill_mode    uplo,
                                                     rocgraph_storage_mode storage)
{
    coo_row_ind.resize(0);
    coo_col_ind.resize(0);
    coo_val.resize(0);

    nnz = 0;
}

template struct rocgraph_matrix_factory_zero<int8_t, int32_t, int32_t>;
template struct rocgraph_matrix_factory_zero<int8_t, int64_t, int32_t>;
template struct rocgraph_matrix_factory_zero<int8_t, int64_t, int64_t>;

template struct rocgraph_matrix_factory_zero<float, int32_t, int32_t>;
template struct rocgraph_matrix_factory_zero<float, int64_t, int32_t>;
template struct rocgraph_matrix_factory_zero<float, int64_t, int64_t>;

template struct rocgraph_matrix_factory_zero<double, int32_t, int32_t>;
template struct rocgraph_matrix_factory_zero<double, int64_t, int32_t>;
template struct rocgraph_matrix_factory_zero<double, int64_t, int64_t>;
