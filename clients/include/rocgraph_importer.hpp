/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph_check.hpp"

#include "rocgraph_matrix_coo.hpp"
#include "rocgraph_matrix_csx.hpp"
#include "rocgraph_matrix_dense.hpp"

#include "rocgraph_type_conversion.hpp"

template <typename X, typename Y>
inline void
    rocgraph_importer_copy_mixed_arrays(size_t size, X* __restrict__ x, const Y* __restrict__ y)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(size_t i = 0; i < size; ++i)
    {
        x[i] = static_cast<X>(y[i]);
    }
}

template <typename U>
rocgraph_status rocgraph_importer_switch_base(size_t              size,
                                              U&                  u,
                                              rocgraph_index_base base,
                                              rocgraph_index_base newbase)
{

    if(base != newbase)
    {
        switch(newbase)
        {
        case rocgraph_index_base_one:
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(size_t i = 0; i < size; ++i)
            {
                ++u[i];
            }
            return rocgraph_status_success;
        }

        case rocgraph_index_base_zero:
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(size_t i = 0; i < size; ++i)
            {
                --u[i];
            }

            return rocgraph_status_success;
        }
        }
        return rocgraph_status_invalid_value;
    }

    return rocgraph_status_success;
}

template <typename IMPL>
class rocgraph_importer
{

public:
    template <typename I = rocgraph_int, typename J = rocgraph_int>
    rocgraph_status
        import_graph_csx(rocgraph_direction* dir, J* m, J* n, I* nnz, rocgraph_index_base* base)
    {
        return static_cast<IMPL&>(*this).import_graph_csx(dir, m, n, nnz, base);
    }

    template <typename T, typename I = rocgraph_int, typename J = rocgraph_int>
    rocgraph_status import_graph_csx(I* ptr, J* ind, T* val)
    {
        return static_cast<IMPL&>(*this).import_graph_csx(ptr, ind, val);
    }

    rocgraph_status import_dense_vector(size_t* size)
    {
        return static_cast<IMPL&>(*this).import_dense_vector(size);
    }
    template <typename T>
    rocgraph_status import_dense_vector(T* data, size_t incy)
    {
        return static_cast<IMPL&>(*this).import_dense_vector(data, incy);
    }

    template <typename I = rocgraph_int>
    rocgraph_status import_dense_matrix(rocgraph_order* order, I* m, I* n)
    {
        return static_cast<IMPL&>(*this).import_dense_matrix(order, m, n);
    }

    template <typename T, typename I = rocgraph_int>
    rocgraph_status import_dense_matrix(T* data, I ld)
    {
        return static_cast<IMPL&>(*this).import_dense_matrix(data, ld);
    }

    template <typename I = rocgraph_int>
    rocgraph_status import_graph_coo(I* m, I* n, int64_t* nnz, rocgraph_index_base* base)
    {
        return static_cast<IMPL&>(*this).import_graph_coo(m, n, nnz, base);
    }

    template <typename T, typename I = rocgraph_int>
    rocgraph_status import_graph_coo(I* row_ind, I* col_ind, T* val)
    {
        return static_cast<IMPL&>(*this).import_graph_coo(row_ind, col_ind, val);
    }

public:
    template <rocgraph_direction DIRECTION,
              typename T,
              typename I = rocgraph_int,
              typename J = rocgraph_int>
    rocgraph_status import(host_csx_matrix<DIRECTION, T, I, J>& csx_)
    {

        //
        // Define
        //
        {
            J                   M;
            J                   N;
            I                   nnz;
            rocgraph_index_base base;
            rocgraph_direction  dir;
            rocgraph_status     status = this->import_graph_csx(&dir, &M, &N, &nnz, &base);
            if(status != rocgraph_status_success)
            {
                return status;
            }
            csx_.define(M, N, nnz, base);
        }

        //
        // Import
        //
        {
            rocgraph_status status = this->import_graph_csx<T, I, J>(csx_.ptr, csx_.ind, csx_.val);
            if(status != rocgraph_status_success)
            {
                return status;
            }
        }
        return rocgraph_status_success;
    }

    template <typename T, typename I = rocgraph_int>
    rocgraph_status import(host_coo_matrix<T, I>& matrix_)
    {
        //
        // Define.
        //
        {
            I                   M, N, nnz;
            rocgraph_index_base base;
            rocgraph_status     status = this->import_graph_coo(&M, &N, &nnz, &base);
            if(status != rocgraph_status_success)
            {
                return status;
            }

            matrix_.define(M, N, nnz, base);
        }

        //
        // Import.
        //
        {
            rocgraph_status status
                = this->import_graph_coo(matrix_.ptr, matrix_.ind, matrix_.val, matrix_.base);
            if(status != rocgraph_status_success)
            {
                return status;
            }
        }

        return rocgraph_status_success;
    }

    template <typename T, typename I = rocgraph_int>
    rocgraph_status import(host_dense_matrix<T, I>& that_)
    {

        //
        // Define
        //
        {
            rocgraph_order  order;
            I               M;
            I               N;
            rocgraph_status status = this->import_dense_matrix(&order, &M, &N);
            if(status != rocgraph_status_success)
            {
                return status;
            }
            that_.define(M, N, order);
        }

        //
        // Import
        //
        {
            rocgraph_status status = this->import_dense_matrix<T, I>(that_.val, that_.ld);
            if(status != rocgraph_status_success)
            {
                return status;
            }
        }
        return rocgraph_status_success;
    }

    template <typename T>
    rocgraph_status import(host_dense_vector<T>& that_)
    {

        //
        // Define
        //
        {
            size_t          M;
            rocgraph_status status = this->import_dense_vector(&M);
            if(status != rocgraph_status_success)
            {
                return status;
            }
            that_.resize(M);
        }

        //
        // Import
        //
        {
            static constexpr size_t ld     = 1;
            rocgraph_status         status = this->import_dense_vector<T>(that_.data(), ld);
            if(status != rocgraph_status_success)
            {
                return status;
            }
        }
        return rocgraph_status_success;
    }
};

inline void missing_file_error_message(const char* filename)
{
    std::cerr << "#" << std::endl;
    std::cerr << "# error:" << std::endl;
    std::cerr << "# cannot open file '" << filename << "'" << std::endl;
    std::cerr << "#" << std::endl;
    std::cerr << "# PLEASE READ CAREFULLY !" << std::endl;
    std::cerr << "#" << std::endl;
    std::cerr << "# What could be the reason of this error: " << std::endl;
    std::cerr
        << "# -1 If you are running the benchmarking application, then the path (or the name) "
           "of the file you have specified might contain a typo."
        << std::endl;
    std::cerr << "# -2 If you are running the benchmarking application, then the file you have "
                 "specified does not exist."
              << std::endl;
    std::cerr
        << "# -3 If you are running the testing application, then it expects to find the file "
           "at the specified location. This means that either you did not download the test "
           "matrices, or you did not specify the location of the folder containing your "
           "files. If you want to specify the location of the folder containing your files, "
           "then you will find the needed information with 'rocgraph-test --help'."
           "If you need to download matrices, then a cmake script "
           "'rocgraph_clientmatrices.cmake' is available from the rocgraph client package."
        << std::endl;
    std::cerr << "#" << std::endl;
    std::cerr
        << "# Examples: 'rocgraph_clientmatrices.cmake -DCMAKE_MATRICES_DIR=<path-of-your-folder>'"
        << std::endl;
    std::cerr << "#           'rocgraph-test --matrices-dir <path-of-your-folder>'" << std::endl;
    std::cerr << "# (or        'export "
                 "ROCGRAPH_CLIENTS_MATRICES_DIR=<path-of-your-folder>;rocgraph-test')"
              << std::endl;
    std::cerr << "#" << std::endl;
    exit(rocgraph_status_internal_error);
}
