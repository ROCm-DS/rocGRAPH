/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef ROCGRAPH_MATRIX_FACTORY_BASE_HPP
#define ROCGRAPH_MATRIX_FACTORY_BASE_HPP

#include "rocgraph.hpp"
#include <vector>

template <typename T, typename I, typename J>
struct rocgraph_matrix_factory_base
{
protected:
    rocgraph_matrix_factory_base() {};

public:
    virtual ~rocgraph_matrix_factory_base() {};

    // @brief Initialize a csr-graph matrix.
    // @param[out] csr_row_ptr vector of offsets.
    // @param[out] csr_col_ind vector of column indices.
    // @param[out] csr_val vector of values.
    // @param[inout] M number of rows.
    // @param[inout] N number of columns.
    // @param[inout] nnz number of non-zeros.
    // @param[in] base base of indices.
    // @param[in] matrix_type what type of matrix to generate.
    // @param[in] uplo fill mode of matrix.

    virtual void init_csr(host_dense_vector<I>& csr_row_ptr,
                          host_dense_vector<J>& csr_col_ind,
                          host_dense_vector<T>& csr_val,
                          J&                    M,
                          J&                    N,
                          I&                    nnz,
                          rocgraph_index_base   base,
                          rocgraph_matrix_type  matrix_type,
                          rocgraph_fill_mode    uplo,
                          rocgraph_storage_mode storage)
        = 0;

    // @brief Initialize a coo-graph matrix.
    // @param[out]   coo_row_ind vector of row indices.
    // @param[out]   coo_col_ind vector of column indices.
    // @param[out]   coo_val vector of values.
    // @param[inout] M number of rows.
    // @param[inout] N number of columns.
    // @param[inout] nnz number of non-zeros.
    // @param[in] base base of indices.
    virtual void init_coo(host_dense_vector<I>& coo_row_ind,
                          host_dense_vector<I>& coo_col_ind,
                          host_dense_vector<T>& coo_val,
                          I&                    M,
                          I&                    N,
                          int64_t&              nnz,
                          rocgraph_index_base   base,
                          rocgraph_matrix_type  matrix_type,
                          rocgraph_fill_mode    uplo,
                          rocgraph_storage_mode storage)
        = 0;
};

#endif // ROCGRAPH_MATRIX_FACTORY_BASE_HPP
