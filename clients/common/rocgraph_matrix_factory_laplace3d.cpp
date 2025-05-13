/*! \file */

// Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_init.hpp"
#include "rocgraph_matrix_utils.hpp"

#include "rocgraph_matrix_factory_laplace3d.hpp"

template <typename T, typename I, typename J>
rocgraph_matrix_factory_laplace3d<T, I, J>::rocgraph_matrix_factory_laplace3d(J dimx,
                                                                              J dimy,
                                                                              J dimz)
    : m_dimx(dimx)
    , m_dimy(dimy)
    , m_dimz(dimz){};

template <typename T, typename I, typename J>
void rocgraph_matrix_factory_laplace3d<T, I, J>::init_csr(host_dense_vector<I>& csr_row_ptr,
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
    switch(matrix_type)
    {
    case rocgraph_matrix_type_symmetric:
    case rocgraph_matrix_type_hermitian:
    case rocgraph_matrix_type_triangular:
    {
        host_dense_vector<I> ptr;
        host_dense_vector<J> ind;
        host_dense_vector<T> val;

        rocgraph_init_csr_laplace3d(
            ptr, ind, val, this->m_dimx, this->m_dimy, this->m_dimz, M, N, nnz, base);

        rocgraph_matrix_utils::host_csrtri(ptr.data(),
                                           ind.data(),
                                           val.data(),
                                           csr_row_ptr,
                                           csr_col_ind,
                                           csr_val,
                                           M,
                                           N,
                                           nnz,
                                           base,
                                           uplo);
        break;
    }
    case rocgraph_matrix_type_general:
    {
        rocgraph_init_csr_laplace3d(csr_row_ptr,
                                    csr_col_ind,
                                    csr_val,
                                    this->m_dimx,
                                    this->m_dimy,
                                    this->m_dimz,
                                    M,
                                    N,
                                    nnz,
                                    base);
        break;
    }
    }

    switch(storage)
    {
    case rocgraph_storage_mode_unsorted:
    {
        rocgraph_matrix_utils::host_csrunsort<T, I, J>(
            csr_row_ptr.data(), csr_col_ind.data(), M, base);
        break;
    }
    case rocgraph_storage_mode_sorted:
    {
        break;
    }
    }
}

template <typename T, typename I, typename J>
void rocgraph_matrix_factory_laplace3d<T, I, J>::init_coo(host_dense_vector<I>& coo_row_ind,
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
    switch(matrix_type)
    {
    case rocgraph_matrix_type_symmetric:
    case rocgraph_matrix_type_hermitian:
    case rocgraph_matrix_type_triangular:
    {
        host_dense_vector<I> row_ind;
        host_dense_vector<I> col_ind;
        host_dense_vector<T> val;

        rocgraph_init_coo_laplace3d(
            row_ind, col_ind, val, this->m_dimx, this->m_dimy, this->m_dimz, M, N, nnz, base);

        rocgraph_matrix_utils::host_cootri(row_ind.data(),
                                           col_ind.data(),
                                           val.data(),
                                           coo_row_ind,
                                           coo_col_ind,
                                           coo_val,
                                           M,
                                           N,
                                           nnz,
                                           base,
                                           uplo);
        break;
    }
    case rocgraph_matrix_type_general:
    {
        rocgraph_init_coo_laplace3d(coo_row_ind,
                                    coo_col_ind,
                                    coo_val,
                                    this->m_dimx,
                                    this->m_dimy,
                                    this->m_dimz,
                                    M,
                                    N,
                                    nnz,
                                    base);
        break;
    }
    }

    switch(storage)
    {
    case rocgraph_storage_mode_unsorted:
    {
        rocgraph_matrix_utils::host_coounsort<T, I>(
            coo_row_ind.data(), coo_col_ind.data(), M, nnz, base);
        break;
    }
    case rocgraph_storage_mode_sorted:
    {
        break;
    }
    }
}

template struct rocgraph_matrix_factory_laplace3d<int8_t, int32_t, int32_t>;
template struct rocgraph_matrix_factory_laplace3d<int8_t, int64_t, int32_t>;
template struct rocgraph_matrix_factory_laplace3d<int8_t, int64_t, int64_t>;

template struct rocgraph_matrix_factory_laplace3d<float, int32_t, int32_t>;
template struct rocgraph_matrix_factory_laplace3d<float, int64_t, int32_t>;
template struct rocgraph_matrix_factory_laplace3d<float, int64_t, int64_t>;

template struct rocgraph_matrix_factory_laplace3d<double, int32_t, int32_t>;
template struct rocgraph_matrix_factory_laplace3d<double, int64_t, int32_t>;
template struct rocgraph_matrix_factory_laplace3d<double, int64_t, int64_t>;
