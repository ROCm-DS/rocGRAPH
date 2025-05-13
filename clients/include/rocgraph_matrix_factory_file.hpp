/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef ROCGRAPH_MATRIX_FACTORY_FILE_HPP
#define ROCGRAPH_MATRIX_FACTORY_FILE_HPP

#include "rocgraph_init.hpp"
#include "rocgraph_matrix_factory_base.hpp"

template <rocgraph_matrix_init MATRIX_INIT,
          typename T,
          typename I = rocgraph_int,
          typename J = rocgraph_int>
struct rocgraph_matrix_factory_file : public rocgraph_matrix_factory_base<T, I, J>
{
private:
    std::string m_filename;
    bool        m_toint;

public:
    explicit rocgraph_matrix_factory_file(const char* filename, bool toint = false);

    virtual void init_csr(host_dense_vector<I>& csr_row_ptr,
                          host_dense_vector<J>& csr_col_ind,
                          host_dense_vector<T>& csr_val,
                          J&                    M,
                          J&                    N,
                          I&                    nnz,
                          rocgraph_index_base   base,
                          rocgraph_matrix_type  matrix_type,
                          rocgraph_fill_mode    uplo,
                          rocgraph_storage_mode storage) override;

    virtual void init_coo(host_dense_vector<I>& coo_row_ind,
                          host_dense_vector<I>& coo_col_ind,
                          host_dense_vector<T>& coo_val,
                          I&                    M,
                          I&                    N,
                          int64_t&              nnz,
                          rocgraph_index_base   base,
                          rocgraph_matrix_type  matrix_type,
                          rocgraph_fill_mode    uplo,
                          rocgraph_storage_mode storage) override;
};

template <typename T, typename I = rocgraph_int, typename J = rocgraph_int>
using rocgraph_matrix_factory_mtx = rocgraph_matrix_factory_file<rocgraph_matrix_file_mtx, T, I, J>;

template <typename T, typename I = rocgraph_int, typename J = rocgraph_int>
using rocgraph_matrix_factory_smtx
    = rocgraph_matrix_factory_file<rocgraph_matrix_file_smtx, T, I, J>;

template <typename T, typename I = rocgraph_int, typename J = rocgraph_int>
using rocgraph_matrix_factory_rocalution
    = rocgraph_matrix_factory_file<rocgraph_matrix_file_rocalution, T, I, J>;

#endif // ROCGRAPH_MATRIX_FACTORY_FILE_HPP
