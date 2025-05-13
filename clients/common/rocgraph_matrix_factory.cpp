/*! \file */

// Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_matrix_factory.hpp"
#include "rocgraph_clients_envariables.hpp"
#include "rocgraph_clients_matrices_dir.hpp"
#include "rocgraph_importer_format_t.hpp"
#include "rocgraph_init.hpp"

static void get_matrix_full_filename(const Arguments& arg_,
                                     const char*      extension_,
                                     std::string&     full_filename_)
{

    if(arg_.timing)
    {
        static constexpr bool use_default = false;
        full_filename_                    = rocgraph_clients_matrices_dir_get(use_default);
        full_filename_ += arg_.filename;
    }
    else
    {
        static constexpr bool use_default = true;
        full_filename_                    = rocgraph_clients_matrices_dir_get(use_default);
        full_filename_ += arg_.filename;
        full_filename_ += extension_;
    }
}

//
// Destructor.
//
template <typename T, typename I, typename J>
rocgraph_matrix_factory<T, I, J>::~rocgraph_matrix_factory()
{
    if(this->m_instance)
    {
        delete this->m_instance;
        this->m_instance = nullptr;
    }
}

//
// Constructor.
//
template <typename T, typename I, typename J>
rocgraph_matrix_factory<T, I, J>::rocgraph_matrix_factory(const Arguments&     arg,
                                                          rocgraph_matrix_init matrix,
                                                          bool                 to_int, // = false
                                                          bool                 full_rank, // = false
                                                          bool                 noseed // = false
                                                          )
    : m_arg(arg)
{
    //
    // FORCE REINIT.
    //
    if(false == noseed)
    {
        rocgraph_seedrand();
    }

    switch(matrix)
    {
    case rocgraph_matrix_random:
    {
        rocgraph_matrix_init_kind matrix_init_kind = arg.matrix_init_kind;
        this->m_instance
            = new rocgraph_matrix_factory_random<T, I, J>(full_rank, to_int, matrix_init_kind);
        break;
    }

    case rocgraph_matrix_laplace_2d:
    {
        this->m_instance = new rocgraph_matrix_factory_laplace2d<T, I, J>(arg.dimx, arg.dimy);
        break;
    }

    case rocgraph_matrix_laplace_3d:
    {
        this->m_instance
            = new rocgraph_matrix_factory_laplace3d<T, I, J>(arg.dimx, arg.dimy, arg.dimz);
        break;
    }

    case rocgraph_matrix_tridiagonal:
    {
        this->m_instance = new rocgraph_matrix_factory_tridiagonal<T, I, J>(arg.l, arg.u);
        break;
    }

    case rocgraph_matrix_pentadiagonal:
    {
        this->m_instance
            = new rocgraph_matrix_factory_pentadiagonal<T, I, J>(arg.ll, arg.l, arg.u, arg.uu);
        break;
    }

    case rocgraph_matrix_file_rocalution:
    {
        std::string full_filename;
        get_matrix_full_filename(
            arg,
            rocgraph_importer_format_t::extension(rocgraph_importer_format_t::rocalution),
            full_filename);

        this->m_instance
            = new rocgraph_matrix_factory_rocalution<T, I, J>(full_filename.c_str(), to_int);
        break;
    }

    case rocgraph_matrix_file_mtx:
    {
        std::string full_filename;
        get_matrix_full_filename(
            arg,
            rocgraph_importer_format_t::extension(rocgraph_importer_format_t::matrixmarket),
            full_filename);
        this->m_instance = new rocgraph_matrix_factory_mtx<T, I, J>(full_filename.c_str());
        break;
    }

    case rocgraph_matrix_file_smtx:
    {
        std::string full_filename;
        get_matrix_full_filename(
            arg,
            rocgraph_importer_format_t::extension(rocgraph_importer_format_t::mlcsr),
            full_filename);
        this->m_instance = new rocgraph_matrix_factory_smtx<T, I, J>(full_filename.c_str());
        break;
    }

    case rocgraph_matrix_zero:
    {
        this->m_instance = new rocgraph_matrix_factory_zero<T, I, J>();
        break;
    }
    }

    if(this->m_instance == nullptr)
    {
        std::cerr << "rocgraph_matrix_factory constructor failed, at line " << __LINE__
                  << std::endl;
        throw(1);
    }
}

//
// Constructor.
//
template <typename T, typename I, typename J>
rocgraph_matrix_factory<T, I, J>::rocgraph_matrix_factory(const Arguments& arg,
                                                          bool             to_int, //  = false,
                                                          bool             full_rank, // = false,
                                                          bool             noseed) //  = false)
    : rocgraph_matrix_factory(arg, arg.matrix, to_int, full_rank, noseed)
{
}

//
// COO
//
template <typename T, typename I, typename J>
void rocgraph_matrix_factory<T, I, J>::init_coo(host_dense_vector<I>& coo_row_ind,
                                                host_dense_vector<I>& coo_col_ind,
                                                host_dense_vector<T>& coo_val,
                                                I&                    M,
                                                I&                    N,
                                                int64_t&              nnz,
                                                rocgraph_index_base   base)
{
    this->m_instance->init_coo(coo_row_ind,
                               coo_col_ind,
                               coo_val,
                               M,
                               N,
                               nnz,
                               base,
                               this->m_arg.matrix_type,
                               this->m_arg.uplo,
                               this->m_arg.storage);
}

template <typename T, typename I, typename J>
void rocgraph_matrix_factory<T, I, J>::init_coo(host_coo_matrix<T, I>& that)
{
    that.base = this->m_arg.baseA;
    that.m    = this->m_arg.M;
    that.n    = this->m_arg.N;
    this->m_instance->init_coo(that.row_ind,
                               that.col_ind,
                               that.val,
                               that.m,
                               that.n,
                               that.nnz,
                               that.base,
                               this->m_arg.matrix_type,
                               this->m_arg.uplo,
                               this->m_arg.storage);
}

template <typename T, typename I, typename J>
void rocgraph_matrix_factory<T, I, J>::init_coo(host_coo_matrix<T, I>& that,
                                                I&                     M,
                                                I&                     N,
                                                rocgraph_index_base    base)
{
    that.base = base;
    that.m    = M;
    that.n    = N;
    this->m_instance->init_coo(that.row_ind,
                               that.col_ind,
                               that.val,
                               that.m,
                               that.n,
                               that.nnz,
                               that.base,
                               this->m_arg.matrix_type,
                               this->m_arg.uplo,
                               this->m_arg.storage);
    M = that.m;
    N = that.n;
}

template <typename T, typename I, typename J>
void rocgraph_matrix_factory<T, I, J>::init_coo(host_coo_matrix<T, I>& that, I& M, I& N)
{
    this->init_coo(that, M, N, this->m_arg.baseA);
}

//
// CSR
//
template <typename T, typename I, typename J>
void rocgraph_matrix_factory<T, I, J>::init_csr(host_dense_vector<I>& csr_row_ptr,
                                                host_dense_vector<J>& csr_col_ind,
                                                host_dense_vector<T>& csr_val,
                                                J&                    m,
                                                J&                    n,
                                                I&                    nnz,
                                                rocgraph_index_base   base)
{
    this->m_instance->init_csr(csr_row_ptr,
                               csr_col_ind,
                               csr_val,
                               m,
                               n,
                               nnz,
                               base,
                               this->m_arg.matrix_type,
                               this->m_arg.uplo,
                               this->m_arg.storage);
}

template <typename T, typename I, typename J>
void rocgraph_matrix_factory<T, I, J>::init_csr(host_csr_matrix<T, I, J>& that)
{
    that.base = this->m_arg.baseA;
    that.m    = this->m_arg.M;
    that.n    = this->m_arg.N;
    this->m_instance->init_csr(that.ptr,
                               that.ind,
                               that.val,
                               that.m,
                               that.n,
                               that.nnz,
                               that.base,
                               this->m_arg.matrix_type,
                               this->m_arg.uplo,
                               this->m_arg.storage);
}

template <typename T, typename I, typename J>
void rocgraph_matrix_factory<T, I, J>::init_csr(host_csr_matrix<T, I, J>& that,
                                                J&                        m,
                                                J&                        n,
                                                rocgraph_index_base       base)
{
    that.base = base;
    that.m    = m;
    that.n    = n;
    this->m_instance->init_csr(that.ptr,
                               that.ind,
                               that.val,
                               that.m,
                               that.n,
                               that.nnz,
                               that.base,
                               this->m_arg.matrix_type,
                               this->m_arg.uplo,
                               this->m_arg.storage);
    m = that.m;
    n = that.n;
}

template <typename T, typename I, typename J>
void rocgraph_matrix_factory<T, I, J>::init_csr(host_csr_matrix<T, I, J>& that, J& m, J& n)
{
    this->init_csr(that, m, n, this->m_arg.baseA);
}

//
// CSC
//
template <typename T, typename I, typename J>
void rocgraph_matrix_factory<T, I, J>::init_csc(host_dense_vector<I>& csc_col_ptr,
                                                host_dense_vector<J>& csc_row_ind,
                                                host_dense_vector<T>& csc_val,
                                                J&                    M,
                                                J&                    N,
                                                I&                    nnz,
                                                rocgraph_index_base   base)
{
    this->m_instance->init_csr(csc_col_ptr,
                               csc_row_ind,
                               csc_val,
                               N,
                               M,
                               nnz,
                               base,
                               this->m_arg.matrix_type,
                               this->m_arg.uplo,
                               this->m_arg.storage);
}

template <typename T, typename I, typename J>
void rocgraph_matrix_factory<T, I, J>::init_csc(host_csc_matrix<T, I, J>& that,
                                                J&                        m,
                                                J&                        n,
                                                rocgraph_index_base       base)
{
    that.base = base;
    this->m_instance->init_csr(that.ptr,
                               that.ind,
                               that.val,
                               n,
                               m,
                               that.nnz,
                               that.base,
                               this->m_arg.matrix_type,
                               this->m_arg.uplo,
                               this->m_arg.storage);
    that.m = m;
    that.n = n;
}

//
// INSTANTIATE.
//
template struct rocgraph_matrix_factory<int8_t, int32_t, int32_t>;
template struct rocgraph_matrix_factory<int8_t, int64_t, int32_t>;
template struct rocgraph_matrix_factory<int8_t, int64_t, int64_t>;

template struct rocgraph_matrix_factory<float, int32_t, int32_t>;
template struct rocgraph_matrix_factory<float, int64_t, int32_t>;
template struct rocgraph_matrix_factory<float, int64_t, int64_t>;

template struct rocgraph_matrix_factory<double, int32_t, int32_t>;
template struct rocgraph_matrix_factory<double, int64_t, int32_t>;
template struct rocgraph_matrix_factory<double, int64_t, int64_t>;
