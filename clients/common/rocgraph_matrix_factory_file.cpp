/*! \file */

// Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_matrix_factory_file.hpp"
#include "rocgraph_import.hpp"
#include "rocgraph_importer_impls.hpp"
#include "rocgraph_matrix_utils.hpp"

template <typename T>
static void apply_toint(host_dense_vector<T>& data)
{
    const size_t size = data.size();
    for(size_t i = 0; i < size; ++i)
    {
        data[i] = std::abs(data[i]);
    }
}

/* ============================================================================================ */
/*! \brief  Read matrix from mtx file in COO format */
template <rocgraph_matrix_init MATRIX_INIT>
struct rocgraph_init_file_traits;

template <>
struct rocgraph_init_file_traits<rocgraph_matrix_file_rocalution>
{
    using importer_t = rocgraph_importer_rocalution;
};

template <>
struct rocgraph_init_file_traits<rocgraph_matrix_file_mtx>
{
    using importer_t = rocgraph_importer_matrixmarket;
};

template <>
struct rocgraph_init_file_traits<rocgraph_matrix_file_smtx>
{
    using importer_t = rocgraph_importer_mlcsr;
};

template <rocgraph_matrix_init MATRIX_INIT>
struct rocgraph_init_file
{
    using importer_t = typename rocgraph_init_file_traits<MATRIX_INIT>::importer_t;

    template <typename... S>
    static inline rocgraph_status import_csr(const char* filename, S&&... s)
    {

        importer_t importer(filename);
        return rocgraph_import_graph_csr(importer, s...);
    }

    template <typename... S>
    static inline rocgraph_status import_coo(const char* filename, S&&... s)
    {
        importer_t importer(filename);
        return rocgraph_import_graph_coo(importer, s...);
    }
};

template <rocgraph_matrix_init MATRIX_INIT, typename T, typename I, typename J>
rocgraph_matrix_factory_file<MATRIX_INIT, T, I, J>::rocgraph_matrix_factory_file(
    const char* filename, bool toint)
    : m_filename(filename)
    , m_toint(toint){};

template <rocgraph_matrix_init MATRIX_INIT, typename T, typename I, typename J>
void rocgraph_matrix_factory_file<MATRIX_INIT, T, I, J>::init_csr(host_dense_vector<I>& csr_row_ptr,
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
    host_dense_vector<I> row_ptr;
    host_dense_vector<J> col_ind;
    host_dense_vector<T> val;

    switch(MATRIX_INIT)
    {
    case rocgraph_matrix_file_rocalution:
    {
        rocgraph_init_csr_rocalution(
            this->m_filename.c_str(), row_ptr, col_ind, val, M, N, nnz, base);
        break;
    }

    case rocgraph_matrix_file_mtx:
    {
        rocgraph_init_csr_mtx(this->m_filename.c_str(), row_ptr, col_ind, val, M, N, nnz, base);
        break;
    }
    case rocgraph_matrix_file_smtx:
    {
        rocgraph_init_csr_smtx(this->m_filename.c_str(), row_ptr, col_ind, val, M, N, nnz, base);
        break;
    }
    }

    switch(matrix_type)
    {
    case rocgraph_matrix_type_general:
    {
        csr_row_ptr = row_ptr;
        csr_col_ind = col_ind;
        csr_val     = val;
        break;
    }
    case rocgraph_matrix_type_symmetric:
    case rocgraph_matrix_type_hermitian:
    case rocgraph_matrix_type_triangular:
    {
        rocgraph_matrix_utils::host_csrtri(row_ptr.data(),
                                           col_ind.data(),
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

    //
    // Apply toint?
    //
    if(this->m_toint)
    {
        apply_toint(csr_val);
    }
}

template <rocgraph_matrix_init MATRIX_INIT, typename T, typename I, typename J>
void rocgraph_matrix_factory_file<MATRIX_INIT, T, I, J>::init_coo(host_dense_vector<I>& coo_row_ind,
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
    host_dense_vector<I> row_ind;
    host_dense_vector<I> col_ind;
    host_dense_vector<T> val;

    switch(MATRIX_INIT)
    {
    case rocgraph_matrix_file_rocalution:
    {
        rocgraph_init_coo_rocalution(
            this->m_filename.c_str(), row_ind, col_ind, val, M, N, nnz, base);

        break;
    }

    case rocgraph_matrix_file_mtx:
    {
        rocgraph_init_coo_mtx(this->m_filename.c_str(), row_ind, col_ind, val, M, N, nnz, base);

        break;
    }

    case rocgraph_matrix_file_smtx:
    {
        rocgraph_init_coo_smtx(this->m_filename.c_str(), row_ind, col_ind, val, M, N, nnz, base);

        break;
    }
    }

    switch(matrix_type)
    {
    case rocgraph_matrix_type_general:
    {
        coo_row_ind = row_ind;
        coo_col_ind = col_ind;
        coo_val     = val;
        break;
    }
    case rocgraph_matrix_type_symmetric:
    case rocgraph_matrix_type_hermitian:
    case rocgraph_matrix_type_triangular:
    {
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

    if(this->m_toint)
    {
        apply_toint(coo_val);
    }
}

template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_mtx, int8_t, int32_t, int32_t>;
template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_mtx, int8_t, int64_t, int32_t>;
template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_mtx, int8_t, int64_t, int64_t>;

template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_mtx, float, int32_t, int32_t>;
template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_mtx, float, int64_t, int32_t>;
template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_mtx, float, int64_t, int64_t>;

template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_mtx, double, int32_t, int32_t>;
template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_mtx, double, int64_t, int32_t>;
template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_mtx, double, int64_t, int64_t>;

template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_rocalution,
                                             int8_t,
                                             int32_t,
                                             int32_t>;
template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_rocalution,
                                             int8_t,
                                             int64_t,
                                             int32_t>;
template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_rocalution,
                                             int8_t,
                                             int64_t,
                                             int64_t>;

template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_rocalution,
                                             float,
                                             int32_t,
                                             int32_t>;
template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_rocalution,
                                             float,
                                             int64_t,
                                             int32_t>;
template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_rocalution,
                                             float,
                                             int64_t,
                                             int64_t>;

template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_rocalution,
                                             double,
                                             int32_t,
                                             int32_t>;
template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_rocalution,
                                             double,
                                             int64_t,
                                             int32_t>;
template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_rocalution,
                                             double,
                                             int64_t,
                                             int64_t>;

template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_smtx, int8_t, int32_t, int32_t>;
template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_smtx, int8_t, int64_t, int32_t>;
template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_smtx, int8_t, int64_t, int64_t>;

template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_smtx, float, int32_t, int32_t>;
template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_smtx, float, int64_t, int32_t>;
template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_smtx, float, int64_t, int64_t>;

template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_smtx, double, int32_t, int32_t>;
template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_smtx, double, int64_t, int32_t>;
template struct rocgraph_matrix_factory_file<rocgraph_matrix_file_smtx, double, int64_t, int64_t>;
