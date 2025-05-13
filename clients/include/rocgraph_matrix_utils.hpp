/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef ROCGRAPH_MATRIX_UTILS_HPP
#define ROCGRAPH_MATRIX_UTILS_HPP

#include "rocgraph.hpp"
#include "rocgraph_matrix.hpp"

static inline const float* get_boost_tol(const float* tol)
{
    return tol;
}

static inline const double* get_boost_tol(const double* tol)
{
    return tol;
}

//
// @brief Utility methods for matrices..
//
struct rocgraph_matrix_utils
{

    //
    // @brief Initialize a host dense matrix with random values.
    // @param[in] that Fill \p that matrix.
    //
    template <typename T>
    static void init(host_dense_matrix<T>& that)
    {
        switch(that.order)
        {
        case rocgraph_order_column:
        {
            rocgraph_init<T>(that, that.m, that.n, that.ld);
            break;
        }

        case rocgraph_order_row:
        {
            //
            // Little trick but the resulting matrix is the transpose of the matrix obtained from rocgraph_order_column.
            // If this poses a problem, we need to refactor rocgraph_init.
            //
            rocgraph_init<T>(that, that.n, that.m, that.ld);
            break;
        }
        }
    }

    //
    // @brief Initialize a host dense matrix with random integer values.
    // @param[in] that Fill \p that matrix.
    //
    template <typename T>
    static void init_exact(host_dense_matrix<T>& that)
    {
        switch(that.order)
        {
        case rocgraph_order_column:
        {
            rocgraph_init_exact<T>(that, that.m, that.n, that.ld);
            break;
        }

        case rocgraph_order_row:
        {
            //
            // Little trick but the resulting matrix is the transpose of the matrix obtained from rocgraph_order_column.
            // If this poses a problem, we need to refactor rocgraph_init_exact.
            //
            rocgraph_init_exact<T>(that, that.n, that.m, that.ld);
            break;
        }
        }
    }

    // Extract lower or upper part of input matrix to form new triangular output matrix
    template <typename T, typename I, typename J>
    static void host_csrtri(const I*              ptr,
                            const J*              ind,
                            const T*              val,
                            host_dense_vector<I>& csr_row_ptr,
                            host_dense_vector<J>& csr_col_ind,
                            host_dense_vector<T>& csr_val,
                            J                     M,
                            J                     N,
                            I&                    nnz,
                            rocgraph_index_base   base,
                            rocgraph_fill_mode    uplo)
    {
        if(M != N)
        {
            std::cerr << "error: host_csrtri only accepts square matrices" << std::endl;
            exit(1);
            return;
        }

        nnz = 0;
        if(uplo == rocgraph_fill_mode_lower)
        {
            for(J i = 0; i < M; i++)
            {
                I start = ptr[i] - base;
                I end   = ptr[i + 1] - base;

                for(I j = start; j < end; j++)
                {
                    if(ind[j] - base <= i)
                    {
                        nnz++;
                    }
                }
            }
        }
        else
        {
            for(J i = 0; i < M; i++)
            {
                I start = ptr[i] - base;
                I end   = ptr[i + 1] - base;

                for(I j = start; j < end; j++)
                {
                    if(ind[j] - base >= i)
                    {
                        nnz++;
                    }
                }
            }
        }

        csr_row_ptr.resize(M + 1, 0);
        csr_col_ind.resize(nnz, 0);
        csr_val.resize(nnz, static_cast<T>(0));

        I index        = 0;
        csr_row_ptr[0] = base;

        if(uplo == rocgraph_fill_mode_lower)
        {
            for(J i = 0; i < M; i++)
            {
                I start = ptr[i] - base;
                I end   = ptr[i + 1] - base;

                for(I j = start; j < end; j++)
                {
                    if(ind[j] - base <= i)
                    {
                        csr_col_ind[index] = ind[j];
                        csr_val[index]     = val[j];
                        index++;
                    }
                }

                csr_row_ptr[i + 1] = index + base;
            }
        }
        else
        {
            for(J i = 0; i < M; i++)
            {
                I start = ptr[i] - base;
                I end   = ptr[i + 1] - base;

                for(I j = start; j < end; j++)
                {
                    if(ind[j] - base >= i)
                    {
                        csr_col_ind[index] = ind[j];
                        csr_val[index]     = val[j];
                        index++;
                    }
                }

                csr_row_ptr[i + 1] = index + base;
            }
        }
    }

    template <typename T, typename I>
    static void host_cootri(const I*              row_ind,
                            const I*              col_ind,
                            const T*              val,
                            host_dense_vector<I>& coo_row_ind,
                            host_dense_vector<I>& coo_col_ind,
                            host_dense_vector<T>& coo_val,
                            I                     M,
                            I                     N,
                            int64_t&              nnz,
                            rocgraph_index_base   base,
                            rocgraph_fill_mode    uplo)
    {
        if(M != N)
        {
            std::cerr << "error: host_cootri only accepts square matrices" << std::endl;
            exit(1);
            return;
        }

        int64_t old_nnz = nnz;
        int64_t new_nnz = 0;
        if(uplo == rocgraph_fill_mode_lower)
        {
            int64_t index = 0;
            for(I i = 0; i < M; i++)
            {
                while(index < nnz && row_ind[index] - base == i)
                {
                    if(col_ind[index] - base <= i)
                    {
                        new_nnz++;
                    }

                    index++;
                }
            }
        }
        else
        {
            int64_t index = 0;
            for(I i = 0; i < M; i++)
            {
                while(index < nnz && row_ind[index] - base == i)
                {
                    if(col_ind[index] - base >= i)
                    {
                        new_nnz++;
                    }

                    index++;
                }
            }
        }

        coo_row_ind.resize(new_nnz, 0);
        coo_col_ind.resize(new_nnz, 0);
        coo_val.resize(new_nnz, static_cast<T>(0));

        nnz = 0;
        if(uplo == rocgraph_fill_mode_lower)
        {
            int64_t index = 0;
            for(I i = 0; i < M; i++)
            {
                while(index < old_nnz && row_ind[index] - base == i)
                {
                    if(col_ind[index] - base <= i)
                    {
                        coo_row_ind[nnz] = row_ind[index];
                        coo_col_ind[nnz] = col_ind[index];
                        coo_val[nnz]     = val[index];
                        nnz++;
                    }

                    index++;
                }
            }
        }
        else
        {
            int64_t index = 0;
            for(I i = 0; i < M; i++)
            {
                while(index < old_nnz && row_ind[index] - base == i)
                {
                    if(col_ind[index] - base >= i)
                    {
                        coo_row_ind[nnz] = row_ind[index];
                        coo_col_ind[nnz] = col_ind[index];
                        coo_val[nnz]     = val[index];
                        nnz++;
                    }

                    index++;
                }
            }
        }
    }

    // Shuffle matrix columns so that the matrix is unsorted
    template <typename T, typename I, typename J>
    static void host_csrunsort(const I* csr_row_ptr, J* csr_col_ind, J M, rocgraph_index_base base)
    {
        for(J i = 0; i < M; i++)
        {
            I start = csr_row_ptr[i] - base;
            I end   = csr_row_ptr[i + 1] - base;

            if(start < end)
            {
                std::random_shuffle(&csr_col_ind[start], &csr_col_ind[end]);
            }
        }
    }

    // Shuffle matrix columns so that the matrix is unsorted
    template <typename T, typename I>
    static void host_coounsort(
        const I* coo_row_ind, I* coo_col_ind, I M, int64_t nnz, rocgraph_index_base base)
    {
        int64_t index = 0;

        for(I i = 0; i < M; i++)
        {
            int64_t start = index;
            while(index < nnz && coo_row_ind[index] - base == i)
            {
                index++;
            }
            int64_t end = index;

            if(start < end)
            {
                std::random_shuffle(&coo_col_ind[start], &coo_col_ind[end]);
            }
        }
    }

    template <typename T, typename I, typename J>
    static rocgraph_status host_csrsym(const host_csr_matrix<T, I, J>& A,
                                       host_csr_matrix<T, I, J>&       symA)
    {
        const auto M    = A.m;
        const auto NNZ  = A.nnz;
        const auto base = A.base;

        if(M != A.n)
        {
            return rocgraph_status_invalid_value;
        }

        //
        // Compute transpose.
        //
        host_csr_matrix<T, I, J> trA(M, M, NNZ, base);
        {
            for(J i = 0; i <= M; ++i)
            {
                trA.ptr[i] = static_cast<I>(0);
            }

            for(J i = 0; i < M; ++i)
            {
                for(I k = A.ptr[i] - base; k < A.ptr[i + 1] - base; ++k)
                {
                    const J j = A.ind[k] - base;
                    trA.ptr[j + 1] += 1;
                }
            }

            for(J i = 2; i <= M; ++i)
            {
                trA.ptr[i] += trA.ptr[i - 1];
            }

            for(J i = 0; i < M; ++i)
            {
                for(I k = A.ptr[i] - base; k < A.ptr[i + 1] - base; ++k)
                {
                    const J j           = A.ind[k] - base;
                    trA.ind[trA.ptr[j]] = i;
                    trA.val[trA.ptr[j]] = A.val[k];
                    ++trA.ptr[j];
                }
            }

            for(J i = M; i > 0; --i)
            {
                trA.ptr[i] = trA.ptr[i - 1];
            }
            trA.ptr[0] = 0;

            if(rocgraph_index_base_one == base)
            {
                for(J i = 0; i <= M; ++i)
                {
                    trA.ptr[i] += base;
                }

                for(I i = 0; i < NNZ; ++i)
                {
                    trA.ind[i] += base;
                }
            }
        }
        //
        // Compute transpose done.
        //

        //
        // Compute (A + A^T) / 2
        //
        host_dense_vector<J> blank(M);
        for(size_t i = 0; i < blank.size(); i++)
        {
            blank[i] = 0;
        }
        host_dense_vector<J> select(M);

        symA.define(M, M, 0, base);

        for(J i = 0; i <= M; ++i)
        {
            symA.ptr[i] = 0;
        }

        for(J i = 0; i < M; ++i)
        {
            //
            // Load row from A.
            //
            J select_n = 0;
            for(I k = A.ptr[i] - base; k < A.ptr[i + 1] - base; ++k)
            {
                J j = A.ind[k] - base;
                if(!blank[j])
                {
                    select[select_n] = j;
                    blank[j]         = ++select_n;
                }
            }

            //
            // Load row from A^T
            //
            for(I k = trA.ptr[i] - trA.base; k < trA.ptr[i + 1] - trA.base; ++k)
            {
                J j = trA.ind[k] - trA.base;
                if(!blank[j])
                {
                    select[select_n] = j;
                    blank[j]         = ++select_n;
                }
            }

            //
            // Reset blank.
            //
            for(J k = 0; k < select_n; ++k)
            {
                blank[select[k]] = 0;
            }

            symA.ptr[i + 1] = select_n;
        }

        for(J i = 2; i <= M; ++i)
        {
            symA.ptr[i] += symA.ptr[i - 1];
        }

        symA.define(M, M, symA.ptr[M], base);

        for(J i = 0; i < M; ++i)
        {
            //
            // Load row from A.
            //
            J select_n = 0;
            for(I k = A.ptr[i] - base; k < A.ptr[i + 1] - base; ++k)
            {
                J j = A.ind[k] - base;
                if(!blank[j])
                {
                    select[select_n] = j;
                    blank[j]         = ++select_n;
                }
            }

            //
            // Load row from A^T
            //
            for(I k = trA.ptr[i] - trA.base; k < trA.ptr[i + 1] - base; ++k)
            {
                J j = trA.ind[k] - trA.base;
                if(!blank[j])
                {
                    select[select_n] = j;
                    blank[j]         = ++select_n;
                }
            }

            std::sort(select.data(), select.data() + select_n);

            for(J k = 0; k < select_n; ++k)
            {
                blank[select[k]] = 0;
            }

            for(J k = 0; k < select_n; ++k)
            {
                symA.ind[symA.ptr[i] + k] = select[k];
            }
        }

        if(rocgraph_index_base_one == base)
        {
            for(J i = 0; i <= M; ++i)
            {
                symA.ptr[i] += base;
            }

            for(I i = 0; i < symA.nnz; ++i)
            {
                symA.ind[i] += base;
            }
        }

        return rocgraph_status_success;
    }
};

#endif // ROCGRAPH_MATRIX_UTILS_HPP
