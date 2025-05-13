/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef ROCGRAPH_MATRIX_FACTORY_HPP
#define ROCGRAPH_MATRIX_FACTORY_HPP

#include "rocgraph_import.hpp"
#include "rocgraph_matrix_utils.hpp"

std::string rocgraph_exepath();

#include "rocgraph_matrix_factory_file.hpp"
#include "rocgraph_matrix_factory_laplace2d.hpp"
#include "rocgraph_matrix_factory_laplace3d.hpp"
#include "rocgraph_matrix_factory_pentadiagonal.hpp"
#include "rocgraph_matrix_factory_random.hpp"
#include "rocgraph_matrix_factory_tridiagonal.hpp"
#include "rocgraph_matrix_factory_zero.hpp"

template <typename T, typename I = rocgraph_int, typename J = rocgraph_int>
struct rocgraph_matrix_factory : public rocgraph_matrix_factory_base<T, I, J>
{
public:
    const Arguments& m_arg;

private:
    rocgraph_matrix_factory_base<T, I, J>* m_instance;

public:
    ~rocgraph_matrix_factory();
    rocgraph_matrix_factory(const Arguments&     arg,
                            rocgraph_matrix_init matrix,
                            bool                 to_int    = false,
                            bool                 full_rank = false,
                            bool                 noseed    = false);

    rocgraph_matrix_factory(const rocgraph_matrix_factory& that)            = delete;
    rocgraph_matrix_factory& operator=(const rocgraph_matrix_factory& that) = delete;
    explicit rocgraph_matrix_factory(const Arguments& arg,
                                     bool             to_int    = false,
                                     bool             full_rank = false,
                                     bool             noseed    = false);

    virtual void init_csr(host_dense_vector<I>& csr_row_ptr,
                          host_dense_vector<J>& csr_col_ind,
                          host_dense_vector<T>& csr_val,
                          J&                    m,
                          J&                    n,
                          I&                    nnz,
                          rocgraph_index_base   base,
                          rocgraph_matrix_type  matrix_type,
                          rocgraph_fill_mode    uplo,
                          rocgraph_storage_mode storage)
    {

        return this->m_instance->init_csr(
            csr_row_ptr, csr_col_ind, csr_val, m, n, nnz, base, matrix_type, uplo, storage);
    }

    virtual void init_coo(host_dense_vector<I>& coo_row_ind,
                          host_dense_vector<I>& coo_col_ind,
                          host_dense_vector<T>& coo_val,
                          I&                    m,
                          I&                    n,
                          int64_t&              nnz,
                          rocgraph_index_base   base,
                          rocgraph_matrix_type  matrix_type,
                          rocgraph_fill_mode    uplo,
                          rocgraph_storage_mode storage)
    {
        return this->m_instance->init_coo(
            coo_row_ind, coo_col_ind, coo_val, m, n, nnz, base, matrix_type, uplo, storage);
    }

    //
    // COO
    //
    void init_coo(host_dense_vector<I>& coo_row_ind,
                  host_dense_vector<I>& coo_col_ind,
                  host_dense_vector<T>& coo_val,
                  I&                    m,
                  I&                    n,
                  int64_t&              nnz,
                  rocgraph_index_base   base);

    void init_coo(host_coo_matrix<T, I>& that);
    void init_coo(host_coo_matrix<T, I>& that, I& m, I& n);
    void init_coo(host_coo_matrix<T, I>& that, I& m, I& n, rocgraph_index_base base);

    //
    // CSR
    //
    void init_csr(host_dense_vector<I>& csr_row_ptr,
                  host_dense_vector<J>& csr_col_ind,
                  host_dense_vector<T>& csr_val,
                  J&                    m,
                  J&                    n,
                  I&                    nnz,
                  rocgraph_index_base   base);

    void init_csr(host_csr_matrix<T, I, J>& that);
    void init_csr(host_csr_matrix<T, I, J>& that, J& m, J& n);
    void init_csr(host_csr_matrix<T, I, J>& that, J& m, J& n, rocgraph_index_base base);

    //
    // CSC
    //
    void init_csc(host_dense_vector<I>& csc_col_ptr,
                  host_dense_vector<J>& csc_row_ind,
                  host_dense_vector<T>& csc_val,
                  J&                    m,
                  J&                    n,
                  I&                    nnz,
                  rocgraph_index_base   base);

    void init_csc(host_csc_matrix<T, I, J>& that, J& m, J& n, rocgraph_index_base base);
};

#endif // ROCGRAPH_MATRIX_FACTORY_HPP
