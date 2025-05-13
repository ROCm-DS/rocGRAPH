/*! \file */

// Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "utility.hpp"

#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * ===========================================================================
 *    level 1 GRAPH
 * ===========================================================================
 */
template <typename I, typename T>
void host_axpby(
    I size, I nnz, T alpha, const T* x_val, const I* x_ind, T beta, T* y, rocgraph_index_base base)
{
    for(I i = 0; i < size; ++i)
    {
        y[i] *= beta;
    }

    for(I i = 0; i < nnz; ++i)
    {
        y[x_ind[i] - base] = std::fma(alpha, x_val[i], y[x_ind[i] - base]);
    }
}

template <typename I, typename X, typename Y, typename T>
void host_doti(
    I nnz, const X* x_val, const I* x_ind, const Y* y, T* result, rocgraph_index_base base)
{
    *result = static_cast<T>(0);

    for(I i = 0; i < nnz; ++i)
    {
        *result = std::fma(y[x_ind[i] - base], x_val[i], *result);
    }
}

template <typename I, typename X, typename Y, typename T>
void host_dotci(
    I nnz, const X* x_val, const I* x_ind, const Y* y, T* result, rocgraph_index_base base)
{
    *result = static_cast<T>(0);

    for(I i = 0; i < nnz; ++i)
    {
        *result = std::fma(rocgraph_conj(x_val[i]), y[x_ind[i] - base], *result);
    }
}

template <typename I, typename T>
void host_gthr(I nnz, const T* y, T* x_val, const I* x_ind, rocgraph_index_base base)
{
    for(I i = 0; i < nnz; ++i)
    {
        x_val[i] = y[x_ind[i] - base];
    }
}

template <typename T>
void host_gthrz(
    rocgraph_int nnz, T* y, T* x_val, const rocgraph_int* x_ind, rocgraph_index_base base)
{
    for(rocgraph_int i = 0; i < nnz; ++i)
    {
        x_val[i]           = y[x_ind[i] - base];
        y[x_ind[i] - base] = static_cast<T>(0);
    }
}

template <typename I, typename T>
void host_roti(
    I nnz, T* x_val, const I* x_ind, T* y, const T* c, const T* s, rocgraph_index_base base)
{
    for(I i = 0; i < nnz; ++i)
    {
        I idx = x_ind[i] - base;

        T xs = x_val[i];
        T ys = y[idx];

        x_val[i] = *c * xs + *s * ys;
        y[idx]   = *c * ys - *s * xs;
    }
}

template <typename I, typename T>
void host_sctr(I nnz, const T* x_val, const I* x_ind, T* y, rocgraph_index_base base)
{
    for(I i = 0; i < nnz; ++i)
    {
        y[x_ind[i] - base] = x_val[i];
    }
}

/*
 * ===========================================================================
 *    level 2 GRAPH
 * ===========================================================================
 */

template <typename T, typename I, typename A, typename X, typename Y>
void host_coomv(rocgraph_operation  trans,
                I                   M,
                I                   N,
                int64_t             nnz,
                T                   alpha,
                const I*            coo_row_ind,
                const I*            coo_col_ind,
                const A*            coo_val,
                const X*            x,
                T                   beta,
                Y*                  y,
                rocgraph_index_base base)
{
    if(trans == rocgraph_operation_none)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(I i = 0; i < M; ++i)
        {
            y[i] *= beta;
        }

        for(int64_t i = 0; i < nnz; ++i)
        {
            y[coo_row_ind[i] - base]
                = std::fma(alpha * coo_val[i], x[coo_col_ind[i] - base], y[coo_row_ind[i] - base]);
        }
    }
    else
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(I i = 0; i < N; ++i)
        {
            y[i] *= beta;
        }

        for(int64_t i = 0; i < nnz; ++i)
        {
            I row = coo_row_ind[i] - base;
            I col = coo_col_ind[i] - base;
            T val
                = (trans == rocgraph_operation_transpose) ? coo_val[i] : rocgraph_conj(coo_val[i]);

            y[col] = std::fma(alpha * val, x[row], y[col]);
        }
    }
}

template <typename T, typename I, typename A, typename X, typename Y>
void host_coomv_aos(rocgraph_operation  trans,
                    I                   M,
                    I                   N,
                    int64_t             nnz,
                    T                   alpha,
                    const I*            coo_ind,
                    const A*            coo_val,
                    const X*            x,
                    T                   beta,
                    Y*                  y,
                    rocgraph_index_base base)
{
    switch(trans)
    {
    case rocgraph_operation_none:
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(I i = 0; i < M; ++i)
        {
            y[i] *= beta;
        }

        for(int64_t i = 0; i < nnz; ++i)
        {
            y[coo_ind[2 * i] - base] = std::fma(
                alpha * coo_val[i], x[coo_ind[2 * i + 1] - base], y[coo_ind[2 * i] - base]);
        }

        break;
    }
    case rocgraph_operation_transpose:
    case rocgraph_operation_conjugate_transpose:
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(I i = 0; i < N; ++i)
        {
            y[i] *= beta;
        }

        for(int64_t i = 0; i < nnz; ++i)
        {
            I row = coo_ind[2 * i] - base;
            I col = coo_ind[2 * i + 1] - base;
            T val
                = (trans == rocgraph_operation_transpose) ? coo_val[i] : rocgraph_conj(coo_val[i]);

            y[col] = std::fma(alpha * val, x[row], y[col]);
        }

        break;
    }
    }
}

template <typename A>
inline A conj_val(A val, bool conj)
{
    return conj ? rocgraph_conj(val) : val;
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
static void host_csrmv_general(rocgraph_operation  trans,
                               J                   M,
                               J                   N,
                               I                   nnz,
                               T                   alpha,
                               const I*            csr_row_ptr,
                               const J*            csr_col_ind,
                               const A*            csr_val,
                               const X*            x,
                               T                   beta,
                               Y*                  y,
                               rocgraph_index_base base,
                               rocgraph_spmv_alg   algo,
                               bool                force_conj)
{
    bool conj = (trans == rocgraph_operation_conjugate_transpose || force_conj);

    if(trans == rocgraph_operation_none)
    {
        if(algo == rocgraph_spmv_alg_csr_stream)
        {
            // Get device properties
            int             dev;
            hipDeviceProp_t prop;

            hipGetDevice(&dev);
            hipGetDeviceProperties(&prop, dev);

            int WF_SIZE;
            J   nnz_per_row = (M == 0) ? 0 : (nnz / M);

            if(nnz_per_row < 4)
                WF_SIZE = 2;
            else if(nnz_per_row < 8)
                WF_SIZE = 4;
            else if(nnz_per_row < 16)
                WF_SIZE = 8;
            else if(nnz_per_row < 32)
                WF_SIZE = 16;
            else if(nnz_per_row < 64 || prop.warpSize == 32)
                WF_SIZE = 32;
            else
                WF_SIZE = 64;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(J i = 0; i < M; ++i)
            {
                I row_begin = csr_row_ptr[i] - base;
                I row_end   = csr_row_ptr[i + 1] - base;

                host_dense_vector<T> sum(WF_SIZE, static_cast<T>(0));

                for(I j = row_begin; j < row_end; j += WF_SIZE)
                {
                    for(int k = 0; k < WF_SIZE; ++k)
                    {
                        if(j + k < row_end)
                        {
                            sum[k] = std::fma(alpha * conj_val(csr_val[j + k], conj),
                                              x[csr_col_ind[j + k] - base],
                                              sum[k]);
                        }
                    }
                }

                for(int j = 1; j < WF_SIZE; j <<= 1)
                {
                    for(int k = 0; k < WF_SIZE - j; ++k)
                    {
                        sum[k] += sum[k + j];
                    }
                }

                if(beta == static_cast<T>(0))
                {
                    y[i] = sum[0];
                }
                else
                {
                    y[i] = std::fma(beta, y[i], sum[0]);
                }
            }
        }
        else
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(J i = 0; i < M; ++i)
            {
                T sum = static_cast<T>(0);
                T err = static_cast<T>(0);

                I row_begin = csr_row_ptr[i] - base;
                I row_end   = csr_row_ptr[i + 1] - base;

                for(I j = row_begin; j < row_end; ++j)
                {
                    T old  = sum;
                    T prod = alpha * conj_val(csr_val[j], conj) * x[csr_col_ind[j] - base];

                    sum = sum + prod;
                    err = (old - (sum - (sum - old))) + (prod - (sum - old)) + err;
                }

                if(beta != static_cast<T>(0))
                {
                    y[i] = std::fma(beta, y[i], sum + err);
                }
                else
                {
                    y[i] = sum + err;
                }
            }
        }
    }
    else
    {
        // Scale y with beta
        for(J i = 0; i < N; ++i)
        {
            y[i] *= beta;
        }

        // Transposed SpMV
        for(J i = 0; i < M; ++i)
        {
            I row_begin = csr_row_ptr[i] - base;
            I row_end   = csr_row_ptr[i + 1] - base;
            T row_val   = alpha * x[i];

            for(I j = row_begin; j < row_end; ++j)
            {
                J col  = csr_col_ind[j] - base;
                A val  = conj_val(csr_val[j], conj);
                y[col] = std::fma(
                    static_cast<T>(val), static_cast<T>(row_val), static_cast<T>(y[col]));
            }
        }
    }
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
static void host_csrmv_symmetric(rocgraph_operation  trans,
                                 J                   M,
                                 J                   N,
                                 I                   nnz,
                                 T                   alpha,
                                 const I*            csr_row_ptr,
                                 const J*            csr_col_ind,
                                 const A*            csr_val,
                                 const X*            x,
                                 T                   beta,
                                 Y*                  y,
                                 rocgraph_index_base base,
                                 rocgraph_spmv_alg   algo,
                                 bool                force_conj)
{
    bool conj = (trans == rocgraph_operation_conjugate_transpose || force_conj);

    if(algo == rocgraph_spmv_alg_csr_stream || trans != rocgraph_operation_none)
    {
        // Get device properties
        int             dev;
        hipDeviceProp_t prop;

        hipGetDevice(&dev);
        hipGetDeviceProperties(&prop, dev);

        int WF_SIZE;
        J   nnz_per_row = (M == 0) ? 0 : (nnz / M);

        if(nnz_per_row < 4)
            WF_SIZE = 2;
        else if(nnz_per_row < 8)
            WF_SIZE = 4;
        else if(nnz_per_row < 16)
            WF_SIZE = 8;
        else if(nnz_per_row < 32)
            WF_SIZE = 16;
        else if(nnz_per_row < 64 || prop.warpSize == 32)
            WF_SIZE = 32;
        else
            WF_SIZE = 64;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(J i = 0; i < M; ++i)
        {
            I row_begin = csr_row_ptr[i] - base;
            I row_end   = csr_row_ptr[i + 1] - base;

            host_dense_vector<T> sum(WF_SIZE, static_cast<T>(0));

            for(I j = row_begin; j < row_end; j += WF_SIZE)
            {
                for(int k = 0; k < WF_SIZE; ++k)
                {
                    if(j + k < row_end)
                    {
                        A val  = conj_val(csr_val[j + k], conj);
                        sum[k] = std::fma(alpha * val, x[csr_col_ind[j + k] - base], sum[k]);
                    }
                }
            }

            for(int j = 1; j < WF_SIZE; j <<= 1)
            {
                for(int k = 0; k < WF_SIZE - j; ++k)
                {
                    sum[k] += sum[k + j];
                }
            }

            if(beta == static_cast<T>(0))
            {
                y[i] = sum[0];
            }
            else
            {
                y[i] = std::fma(beta, y[i], sum[0]);
            }
        }

        for(J i = 0; i < M; i++)
        {
            I row_begin = csr_row_ptr[i] - base;
            I row_end   = csr_row_ptr[i + 1] - base;

            T x_val = alpha * x[i];
            for(I j = row_begin; j < row_end; ++j)
            {
                if((csr_col_ind[j] - base) != i)
                {
                    y[csr_col_ind[j] - base] = std::fma(static_cast<T>(conj_val(csr_val[j], conj)),
                                                        static_cast<T>(x_val),
                                                        static_cast<T>(y[csr_col_ind[j] - base]));
                }
            }
        }
    }
    else
    {
        // Scale y with beta
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(J i = 0; i < M; ++i)
        {
            y[i] *= beta;
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(J i = 0; i < M; i++)
        {
            T sum = static_cast<T>(0);
            T err = static_cast<T>(0);

            I row_begin = csr_row_ptr[i] - base;
            I row_end   = csr_row_ptr[i + 1] - base;

            for(I j = row_begin; j < row_end; ++j)
            {
                T old  = sum;
                T prod = alpha * conj_val(csr_val[j], conj) * x[csr_col_ind[j] - base];

                sum = sum + prod;
                err = (old - (sum - (sum - old))) + (prod - (sum - old)) + err;
            }

            y[i] += sum + err;
        }

        for(J i = 0; i < M; i++)
        {
            I row_begin = csr_row_ptr[i] - base;
            I row_end   = csr_row_ptr[i + 1] - base;

            T x_val = alpha * x[i];
            for(I j = row_begin; j < row_end; ++j)
            {
                if((csr_col_ind[j] - base) != i)
                {
                    y[csr_col_ind[j] - base] = std::fma(static_cast<T>(conj_val(csr_val[j], conj)),
                                                        static_cast<T>(x_val),
                                                        static_cast<T>(y[csr_col_ind[j] - base]));
                }
            }
        }
    }
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
void host_csrmv(rocgraph_operation   trans,
                J                    M,
                J                    N,
                I                    nnz,
                T                    alpha,
                const I*             csr_row_ptr,
                const J*             csr_col_ind,
                const A*             csr_val,
                const X*             x,
                T                    beta,
                Y*                   y,
                rocgraph_index_base  base,
                rocgraph_matrix_type matrix_type,
                rocgraph_spmv_alg    algo,
                bool                 force_conj)
{
    switch(matrix_type)
    {
    case rocgraph_matrix_type_symmetric:
    {
        host_csrmv_symmetric(trans,
                             M,
                             N,
                             nnz,
                             alpha,
                             csr_row_ptr,
                             csr_col_ind,
                             csr_val,
                             x,
                             beta,
                             y,
                             base,
                             algo,
                             force_conj);
        break;
    }
    default:
    {
        host_csrmv_general(trans,
                           M,
                           N,
                           nnz,
                           alpha,
                           csr_row_ptr,
                           csr_col_ind,
                           csr_val,
                           x,
                           beta,
                           y,
                           base,
                           algo,
                           force_conj);
        break;
    }
    }
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
void host_cscmv(rocgraph_operation trans,
                J                  M,
                J                  N,
                I                  nnz,
                T                  alpha,
                const I* __restrict csc_col_ptr,
                const J* __restrict csc_row_ind,
                const A* __restrict csc_val,
                const X* __restrict x,
                T beta,
                Y* __restrict y,
                rocgraph_index_base  base,
                rocgraph_matrix_type matrix_type,
                rocgraph_spmv_alg    algo)
{
    switch(trans)
    {
    case rocgraph_operation_none:
    {
        return host_csrmv(rocgraph_operation_transpose,
                          N,
                          M,
                          nnz,
                          alpha,
                          csc_col_ptr,
                          csc_row_ind,
                          csc_val,
                          x,
                          beta,
                          y,
                          base,
                          matrix_type,
                          algo,
                          false);
    }
    case rocgraph_operation_transpose:
    {
        return host_csrmv(rocgraph_operation_none,
                          N,
                          M,
                          nnz,
                          alpha,
                          csc_col_ptr,
                          csc_row_ind,
                          csc_val,
                          x,
                          beta,
                          y,
                          base,
                          matrix_type,
                          algo,
                          false);
    }
    case rocgraph_operation_conjugate_transpose:
    {
        return host_csrmv(rocgraph_operation_none,
                          N,
                          M,
                          nnz,
                          alpha,
                          csc_col_ptr,
                          csc_row_ind,
                          csc_val,
                          x,
                          beta,
                          y,
                          base,
                          matrix_type,
                          algo,
                          true);
    }
    }
}

template <typename I, typename J, typename T>
static void host_csr_lsolve(J                   M,
                            T                   alpha,
                            const I*            csr_row_ptr,
                            const J*            csr_col_ind,
                            const T*            csr_val,
                            const T*            x,
                            int64_t             x_inc,
                            T*                  y,
                            rocgraph_diag_type  diag_type,
                            rocgraph_index_base base,
                            J*                  struct_pivot,
                            J*                  numeric_pivot)
{
    // Get device properties
    int             dev;
    hipDeviceProp_t prop;

    hipGetDevice(&dev);
    hipGetDeviceProperties(&prop, dev);

    host_dense_vector<T> temp(prop.warpSize);

    // Process lower triangular part
    for(J row = 0; row < M; ++row)
    {
        temp.assign(prop.warpSize, static_cast<T>(0));
        temp[0] = alpha * x[x_inc * row];

        I diag      = -1;
        I row_begin = csr_row_ptr[row] - base;
        I row_end   = csr_row_ptr[row + 1] - base;

        T diag_val = static_cast<T>(0);

        for(I l = row_begin; l < row_end; l += prop.warpSize)
        {
            for(uint32_t k = 0; k < prop.warpSize; ++k)
            {
                I j = l + k;

                // Do not run out of bounds
                if(j >= row_end)
                {
                    break;
                }

                J local_col = csr_col_ind[j] - base;
                T local_val = csr_val[j];

                if(local_val == static_cast<T>(0) && local_col == row
                   && diag_type == rocgraph_diag_type_non_unit)
                {
                    // Numerical zero pivot found, avoid division by 0
                    // and store index for later use.
                    *numeric_pivot = std::min(*numeric_pivot, row + base);
                    local_val      = static_cast<T>(1);
                }

                // Ignore all entries that are above the diagonal
                if(local_col > row)
                {
                    break;
                }

                // Diagonal entry
                if(local_col == row)
                {
                    // If diagonal type is non unit, do division by diagonal entry
                    // This is not required for unit diagonal for obvious reasons
                    if(diag_type == rocgraph_diag_type_non_unit)
                    {
                        diag     = j;
                        diag_val = static_cast<T>(1) / local_val;
                    }

                    break;
                }

                // Lower triangular part
                temp[k] = std::fma(-local_val, y[local_col], temp[k]);
            }
        }

        for(uint32_t j = 1; j < prop.warpSize; j <<= 1)
        {
            for(uint32_t k = 0; k < prop.warpSize - j; ++k)
            {
                temp[k] += temp[k + j];
            }
        }

        if(diag_type == rocgraph_diag_type_non_unit)
        {
            if(diag == -1)
            {
                *struct_pivot = std::min(*struct_pivot, row + base);
            }

            y[row] = temp[0] * diag_val;
        }
        else
        {
            y[row] = temp[0];
        }
    }
}

template <typename I, typename J, typename T>
static void host_csr_usolve(J                   M,
                            T                   alpha,
                            const I*            csr_row_ptr,
                            const J*            csr_col_ind,
                            const T*            csr_val,
                            const T*            x,
                            int64_t             x_inc,
                            T*                  y,
                            rocgraph_diag_type  diag_type,
                            rocgraph_index_base base,
                            J*                  struct_pivot,
                            J*                  numeric_pivot)
{

    // Get device properties
    int             dev;
    hipDeviceProp_t prop;

    hipGetDevice(&dev);
    hipGetDeviceProperties(&prop, dev);

    host_dense_vector<T> temp(prop.warpSize);

    // Process upper triangular part
    for(J row = M - 1; row >= 0; --row)
    {

        temp.assign(prop.warpSize, static_cast<T>(0));
        temp[0] = alpha * x[x_inc * row];

        I diag      = -1;
        I row_begin = csr_row_ptr[row] - base;
        I row_end   = csr_row_ptr[row + 1] - base;

        T diag_val = static_cast<T>(0);

        for(I l = row_end - 1; l >= row_begin; l -= prop.warpSize)
        {
            for(uint32_t k = 0; k < prop.warpSize; ++k)
            {
                I j = l - k;

                // Do not run out of bounds
                if(j < row_begin)
                {
                    break;
                }

                J local_col = csr_col_ind[j] - base;
                T local_val = csr_val[j];

                // Ignore all entries that are below the diagonal
                if(local_col < row)
                {
                    continue;
                }

                // Diagonal entry
                if(local_col == row)
                {
                    if(diag_type == rocgraph_diag_type_non_unit)
                    {
                        // Check for numerical zero
                        if(local_val == static_cast<T>(0))
                        {
                            *numeric_pivot = std::min(*numeric_pivot, row + base);
                            local_val      = static_cast<T>(1);
                        }

                        diag     = j;
                        diag_val = static_cast<T>(1) / local_val;
                    }

                    continue;
                }

                // Upper triangular part
                temp[k] = std::fma(-local_val, y[local_col], temp[k]);
            }
        }

        for(uint32_t j = 1; j < prop.warpSize; j <<= 1)
        {
            for(uint32_t k = 0; k < prop.warpSize - j; ++k)
            {
                temp[k] += temp[k + j];
            }
        }

        if(diag_type == rocgraph_diag_type_non_unit)
        {
            if(diag == -1)
            {
                *struct_pivot = std::min(*struct_pivot, row + base);
            }

            y[row] = temp[0] * diag_val;
        }
        else
        {
            y[row] = temp[0];
        }
    }
}

template <typename I, typename J, typename T>
void host_csrsv(rocgraph_operation  trans,
                J                   M,
                I                   nnz,
                T                   alpha,
                const I*            csr_row_ptr,
                const J*            csr_col_ind,
                const T*            csr_val,
                const T*            x,
                int64_t             x_inc,
                T*                  y,
                rocgraph_diag_type  diag_type,
                rocgraph_fill_mode  fill_mode,
                rocgraph_index_base base,
                J*                  struct_pivot,
                J*                  numeric_pivot)
{

    // Initialize pivot
    *struct_pivot  = M + 1;
    *numeric_pivot = M + 1;

    if(trans == rocgraph_operation_none)
    {
        if(fill_mode == rocgraph_fill_mode_lower)
        {
            host_csr_lsolve(M,
                            alpha,
                            csr_row_ptr,
                            csr_col_ind,
                            csr_val,
                            x,
                            x_inc,
                            y,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
        else
        {
            host_csr_usolve(M,
                            alpha,
                            csr_row_ptr,
                            csr_col_ind,
                            csr_val,
                            x,
                            x_inc,
                            y,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
    }
    else if(trans == rocgraph_operation_transpose
            || trans == rocgraph_operation_conjugate_transpose)
    {
        // Transpose matrix
        host_dense_vector<I> csrt_row_ptr(M + 1);
        host_dense_vector<J> csrt_col_ind(nnz);
        host_dense_vector<T> csrt_val(nnz);

        host_csr_to_csc(M,
                        M,
                        nnz,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        csrt_col_ind,
                        csrt_row_ptr,
                        csrt_val,
                        rocgraph_action_numeric,
                        base);

        if(trans == rocgraph_operation_conjugate_transpose)
        {
            for(size_t i = 0; i < csrt_val.size(); i++)
            {
                csrt_val[i] = rocgraph_conj(csrt_val[i]);
            }
        }

        if(fill_mode == rocgraph_fill_mode_lower)
        {
            host_csr_usolve(M,
                            alpha,
                            csrt_row_ptr.data(),
                            csrt_col_ind.data(),
                            csrt_val.data(),
                            x,
                            x_inc,
                            y,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
        else
        {
            host_csr_lsolve(M,
                            alpha,
                            csrt_row_ptr.data(),
                            csrt_col_ind.data(),
                            csrt_val.data(),
                            x,
                            x_inc,
                            y,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
    }

    *numeric_pivot = std::min(*numeric_pivot, *struct_pivot);

    *struct_pivot  = (*struct_pivot == M + 1) ? -1 : *struct_pivot;
    *numeric_pivot = (*numeric_pivot == M + 1) ? -1 : *numeric_pivot;
}

template <typename I, typename T>
void host_coosv(rocgraph_operation  trans,
                I                   M,
                int64_t             nnz,
                T                   alpha,
                const I*            coo_row_ind,
                const I*            coo_col_ind,
                const T*            coo_val,
                const T*            x,
                T*                  y,
                rocgraph_diag_type  diag_type,
                rocgraph_fill_mode  fill_mode,
                rocgraph_index_base base,
                I*                  struct_pivot,
                I*                  numeric_pivot)
{
    if(std::is_same<I, int32_t>() && nnz < std::numeric_limits<int32_t>::max())
    {
        host_dense_vector<int32_t> csr_row_ptr(M + 1);

        host_coo_to_csr<int32_t, I>(M, nnz, coo_row_ind, csr_row_ptr.data(), base);

        host_csrsv<int32_t, I>(trans,
                               M,
                               nnz,
                               alpha,
                               csr_row_ptr.data(),
                               coo_col_ind,
                               coo_val,
                               x,
                               (int64_t)1,
                               y,
                               diag_type,
                               fill_mode,
                               base,
                               struct_pivot,
                               numeric_pivot);
    }
    else
    {
        host_dense_vector<int64_t> csr_row_ptr(M + 1);

        host_coo_to_csr(M, nnz, coo_row_ind, csr_row_ptr.data(), base);

        host_csrsv(trans,
                   M,
                   nnz,
                   alpha,
                   csr_row_ptr.data(),
                   coo_col_ind,
                   coo_val,
                   x,
                   (int64_t)1,
                   y,
                   diag_type,
                   fill_mode,
                   base,
                   struct_pivot,
                   numeric_pivot);
    }
}

template <typename T, typename I, typename A, typename X, typename Y>
void host_ellmv(rocgraph_operation  trans,
                I                   M,
                I                   N,
                T                   alpha,
                const I*            ell_col_ind,
                const A*            ell_val,
                I                   ell_width,
                const X*            x,
                T                   beta,
                Y*                  y,
                rocgraph_index_base base)
{
    if(trans == rocgraph_operation_none)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(I i = 0; i < M; ++i)
        {
            T sum = static_cast<T>(0);
            for(I p = 0; p < ell_width; ++p)
            {
                int64_t idx = (int64_t)p * M + i;
                I       col = ell_col_ind[idx] - base;

                if(col >= 0 && col < N)
                {
                    sum = std::fma(
                        static_cast<T>(ell_val[idx]), static_cast<T>(x[col]), static_cast<T>(sum));
                }
                else
                {
                    break;
                }
            }

            if(beta != static_cast<T>(0))
            {
                y[i] = std::fma(
                    static_cast<T>(beta), static_cast<T>(y[i]), static_cast<T>(alpha * sum));
            }
            else
            {
                y[i] = alpha * sum;
            }
        }
    }
    else
    {
        // Scale y with beta
        for(I i = 0; i < N; ++i)
        {
            y[i] *= beta;
        }

        // Transposed SpMV
        for(I i = 0; i < M; ++i)
        {
            T row_val = alpha * x[i];

            for(I p = 0; p < ell_width; ++p)
            {
                int64_t idx = (int64_t)p * M + i;
                I       col = ell_col_ind[idx] - base;

                if(col >= 0 && col < N)
                {
                    T val = (trans == rocgraph_operation_conjugate_transpose)
                                ? rocgraph_conj(ell_val[idx])
                                : ell_val[idx];

                    y[col] = std::fma(
                        static_cast<T>(val), static_cast<T>(row_val), static_cast<T>(y[col]));
                }
                else
                {
                    break;
                }
            }
        }
    }
}

template <typename T>
void host_hybmv(rocgraph_operation  trans,
                rocgraph_int        M,
                rocgraph_int        N,
                T                   alpha,
                rocgraph_int        ell_nnz,
                const rocgraph_int* ell_col_ind,
                const T*            ell_val,
                rocgraph_int        ell_width,
                rocgraph_int        coo_nnz,
                const rocgraph_int* coo_row_ind,
                const rocgraph_int* coo_col_ind,
                const T*            coo_val,
                const T*            x,
                T                   beta,
                T*                  y,
                rocgraph_index_base base)
{
    T coo_beta = beta;

    // ELL part
    if(ell_nnz > 0)
    {
        host_ellmv(trans, M, N, alpha, ell_col_ind, ell_val, ell_width, x, beta, y, base);
        coo_beta = static_cast<T>(1);
    }

    // COO part
    if(coo_nnz > 0)
    {
        host_coomv(
            trans, M, N, coo_nnz, alpha, coo_row_ind, coo_col_ind, coo_val, x, coo_beta, y, base);
    }
}

/*
 * ===========================================================================
 *    level 3 GRAPH
 * ===========================================================================
 */

template <typename T, typename I, typename J>
void host_csrmm(J                   M,
                J                   N,
                J                   K,
                rocgraph_operation  transA,
                rocgraph_operation  transB,
                T                   alpha,
                const I*            csr_row_ptr_A,
                const J*            csr_col_ind_A,
                const T*            csr_val_A,
                const T*            B,
                int64_t             ldb,
                rocgraph_order      order_B,
                T                   beta,
                T*                  C,
                int64_t             ldc,
                rocgraph_order      order_C,
                rocgraph_index_base base,
                bool                force_conj_A)
{
    bool conj_A = (transA == rocgraph_operation_conjugate_transpose || force_conj_A);
    bool conj_B = (transB == rocgraph_operation_conjugate_transpose);

    if(transA == rocgraph_operation_none)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(J i = 0; i < M; i++)
        {
            I row_begin = csr_row_ptr_A[i] - base;
            I row_end   = csr_row_ptr_A[i + 1] - base;

            for(J j = 0; j < N; ++j)
            {
                T sum = static_cast<T>(0);

                for(I k = row_begin; k < row_end; ++k)
                {
                    int64_t idx_B = 0;
                    if((transB == rocgraph_operation_none && order_B == rocgraph_order_column)
                       || (transB == rocgraph_operation_transpose && order_B == rocgraph_order_row)
                       || (transB == rocgraph_operation_conjugate_transpose
                           && order_B == rocgraph_order_row))
                    {
                        idx_B = (csr_col_ind_A[k] - base + j * ldb);
                    }
                    else
                    {
                        idx_B = (j + (csr_col_ind_A[k] - base) * ldb);
                    }

                    sum = std::fma(conj_val(csr_val_A[k], conj_A), conj_val(B[idx_B], conj_B), sum);
                }

                int64_t idx_C = (order_C == rocgraph_order_column) ? i + j * ldc : i * ldc + j;

                if(beta == static_cast<T>(0))
                {
                    C[idx_C] = alpha * sum;
                }
                else
                {
                    C[idx_C] = std::fma(beta, C[idx_C], alpha * sum);
                }
            }
        }
    }
    else
    {
        // scale C by beta
        for(J i = 0; i < K; i++)
        {
            for(J j = 0; j < N; ++j)
            {
                int64_t idx_C = (order_C == rocgraph_order_column) ? i + j * ldc : i * ldc + j;
                C[idx_C]      = beta * C[idx_C];
            }
        }

        for(J i = 0; i < M; i++)
        {
            I row_begin = csr_row_ptr_A[i] - base;
            I row_end   = csr_row_ptr_A[i + 1] - base;

            for(J j = 0; j < N; ++j)
            {
                for(I k = row_begin; k < row_end; ++k)
                {
                    J col = csr_col_ind_A[k] - base;
                    T val = conj_val(csr_val_A[k], conj_A);

                    int64_t idx_B = 0;

                    if((transB == rocgraph_operation_none && order_B == rocgraph_order_column)
                       || (transB == rocgraph_operation_transpose && order_B == rocgraph_order_row)
                       || (transB == rocgraph_operation_conjugate_transpose
                           && order_B == rocgraph_order_row))
                    {
                        idx_B = (i + j * ldb);
                    }
                    else
                    {
                        idx_B = (j + i * ldb);
                    }

                    int64_t idx_C
                        = (order_C == rocgraph_order_column) ? col + j * ldc : col * ldc + j;

                    C[idx_C] += alpha * val * conj_val(B[idx_B], conj_B);
                }
            }
        }
    }
}

template <typename T, typename I, typename J>
void host_csrmm_batched(J                   M,
                        J                   N,
                        J                   K,
                        J                   batch_count_A,
                        int64_t             offsets_batch_stride_A,
                        int64_t             columns_values_batch_stride_A,
                        rocgraph_operation  transA,
                        rocgraph_operation  transB,
                        T                   alpha,
                        const I*            csr_row_ptr_A,
                        const J*            csr_col_ind_A,
                        const T*            csr_val_A,
                        const T*            B,
                        int64_t             ldb,
                        J                   batch_count_B,
                        int64_t             batch_stride_B,
                        rocgraph_order      order_B,
                        T                   beta,
                        T*                  C,
                        int64_t             ldc,
                        J                   batch_count_C,
                        int64_t             batch_stride_C,
                        rocgraph_order      order_C,
                        rocgraph_index_base base,
                        bool                force_conj_A)
{
    const bool Ci_A_Bi  = (batch_count_A == 1 && batch_count_B == batch_count_C);
    const bool Ci_Ai_B  = (batch_count_B == 1 && batch_count_A == batch_count_C);
    const bool Ci_Ai_Bi = (batch_count_A == batch_count_C && batch_count_A == batch_count_B);

    if(!Ci_A_Bi && !Ci_Ai_B && !Ci_Ai_Bi)
    {
        return;
    }

    if(Ci_A_Bi)
    {
        for(J i = 0; i < batch_count_C; i++)
        {
            host_csrmm(M,
                       N,
                       K,
                       transA,
                       transB,
                       alpha,
                       csr_row_ptr_A,
                       csr_col_ind_A,
                       csr_val_A,
                       B + batch_stride_B * i,
                       ldb,
                       order_B,
                       beta,
                       C + batch_stride_C * i,
                       ldc,
                       order_C,
                       base,
                       force_conj_A);
        }
    }
    else if(Ci_Ai_B)
    {
        for(J i = 0; i < batch_count_C; i++)
        {
            host_csrmm(M,
                       N,
                       K,
                       transA,
                       transB,
                       alpha,
                       csr_row_ptr_A + offsets_batch_stride_A * i,
                       csr_col_ind_A + columns_values_batch_stride_A * i,
                       csr_val_A + columns_values_batch_stride_A * i,
                       B,
                       ldb,
                       order_B,
                       beta,
                       C + batch_stride_C * i,
                       ldc,
                       order_C,
                       base,
                       force_conj_A);
        }
    }
    else if(Ci_Ai_Bi)
    {
        for(J i = 0; i < batch_count_C; i++)
        {
            host_csrmm(M,
                       N,
                       K,
                       transA,
                       transB,
                       alpha,
                       csr_row_ptr_A + offsets_batch_stride_A * i,
                       csr_col_ind_A + columns_values_batch_stride_A * i,
                       csr_val_A + columns_values_batch_stride_A * i,
                       B + batch_stride_B * i,
                       ldb,
                       order_B,
                       beta,
                       C + batch_stride_C * i,
                       ldc,
                       order_C,
                       base,
                       force_conj_A);
        }
    }
}

template <typename T, typename I>
void host_coomm(I                   M,
                I                   N,
                I                   K,
                int64_t             nnz,
                rocgraph_operation  transA,
                rocgraph_operation  transB,
                T                   alpha,
                const I*            coo_row_ind_A,
                const I*            coo_col_ind_A,
                const T*            coo_val_A,
                const T*            B,
                int64_t             ldb,
                rocgraph_order      order_B,
                T                   beta,
                T*                  C,
                int64_t             ldc,
                rocgraph_order      order_C,
                rocgraph_index_base base)
{
    bool conj_A = (transA == rocgraph_operation_conjugate_transpose);
    bool conj_B = (transB == rocgraph_operation_conjugate_transpose);

    if(transA == rocgraph_operation_none)
    {
        for(I j = 0; j < N; j++)
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(I i = 0; i < M; ++i)
            {
                int64_t idx_C = (order_C == rocgraph_order_column) ? i + j * ldc : i * ldc + j;
                C[idx_C] *= beta;
            }
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(I j = 0; j < N; j++)
        {
            for(int64_t i = 0; i < nnz; ++i)
            {
                I row = coo_row_ind_A[i] - base;
                I col = coo_col_ind_A[i] - base;
                T val = alpha * coo_val_A[i];

                int64_t idx_C = (order_C == rocgraph_order_column) ? row + j * ldc : row * ldc + j;

                int64_t idx_B = 0;
                if((transB == rocgraph_operation_none && order_B == rocgraph_order_column)
                   || (transB != rocgraph_operation_none && order_B != rocgraph_order_column))
                {
                    idx_B = (col + j * ldb);
                }
                else
                {
                    idx_B = (j + col * ldb);
                }

                C[idx_C] = std::fma(val, conj_val(B[idx_B], conj_B), C[idx_C]);
            }
        }
    }
    else
    {
        for(I j = 0; j < N; j++)
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(I i = 0; i < K; ++i)
            {
                int64_t idx_C = (order_C == rocgraph_order_column) ? i + j * ldc : i * ldc + j;
                C[idx_C] *= beta;
            }
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(I j = 0; j < N; j++)
        {
            for(int64_t i = 0; i < nnz; ++i)
            {
                I row = coo_row_ind_A[i] - base;
                I col = coo_col_ind_A[i] - base;
                T val = alpha * conj_val(coo_val_A[i], conj_A);

                int64_t idx_C = (order_C == rocgraph_order_column) ? col + j * ldc : col * ldc + j;

                int64_t idx_B = 0;
                if((transB == rocgraph_operation_none && order_B == rocgraph_order_column)
                   || (transB != rocgraph_operation_none && order_B != rocgraph_order_column))
                {
                    idx_B = (row + j * ldb);
                }
                else
                {
                    idx_B = (j + row * ldb);
                }

                C[idx_C] = std::fma(val, conj_val(B[idx_B], conj_B), C[idx_C]);
            }
        }
    }
}

template <typename T, typename I>
void host_coomm_batched(I                   M,
                        I                   N,
                        I                   K,
                        int64_t             nnz,
                        I                   batch_count_A,
                        int64_t             batch_stride_A,
                        rocgraph_operation  transA,
                        rocgraph_operation  transB,
                        T                   alpha,
                        const I*            coo_row_ind_A,
                        const I*            coo_col_ind_A,
                        const T*            coo_val_A,
                        const T*            B,
                        int64_t             ldb,
                        I                   batch_count_B,
                        int64_t             batch_stride_B,
                        rocgraph_order      order_B,
                        T                   beta,
                        T*                  C,
                        int64_t             ldc,
                        I                   batch_count_C,
                        int64_t             batch_stride_C,
                        rocgraph_order      order_C,
                        rocgraph_index_base base)
{
    bool Ci_A_Bi  = (batch_count_A == 1 && batch_count_B == batch_count_C);
    bool Ci_Ai_B  = (batch_count_B == 1 && batch_count_A == batch_count_C);
    bool Ci_Ai_Bi = (batch_count_A == batch_count_C && batch_count_A == batch_count_B);

    if(!Ci_A_Bi && !Ci_Ai_B && !Ci_Ai_Bi)
    {
        return;
    }

    if(Ci_A_Bi)
    {
        for(I i = 0; i < batch_count_C; i++)
        {
            host_coomm(M,
                       N,
                       K,
                       nnz,
                       transA,
                       transB,
                       alpha,
                       coo_row_ind_A,
                       coo_col_ind_A,
                       coo_val_A,
                       B + batch_stride_B * i,
                       ldb,
                       order_B,
                       beta,
                       C + batch_stride_C * i,
                       ldc,
                       order_C,
                       base);
        }
    }
    else if(Ci_Ai_B)
    {
        for(I i = 0; i < batch_count_C; i++)
        {
            host_coomm(M,
                       N,
                       K,
                       nnz,
                       transA,
                       transB,
                       alpha,
                       coo_row_ind_A + batch_stride_A * i,
                       coo_col_ind_A + batch_stride_A * i,
                       coo_val_A + batch_stride_A * i,
                       B,
                       ldb,
                       order_B,
                       beta,
                       C + batch_stride_C * i,
                       ldc,
                       order_C,
                       base);
        }
    }
    else if(Ci_Ai_Bi)
    {
        for(I i = 0; i < batch_count_C; i++)
        {
            host_coomm(M,
                       N,
                       K,
                       nnz,
                       transA,
                       transB,
                       alpha,
                       coo_row_ind_A + batch_stride_A * i,
                       coo_col_ind_A + batch_stride_A * i,
                       coo_val_A + batch_stride_A * i,
                       B + batch_stride_B * i,
                       ldb,
                       order_B,
                       beta,
                       C + batch_stride_C * i,
                       ldc,
                       order_C,
                       base);
        }
    }
}

template <typename T, typename I, typename J>
void host_cscmm(J                  M,
                J                  N,
                J                  K,
                rocgraph_operation transA,
                rocgraph_operation transB,
                T                  alpha,
                const I* __restrict csc_col_ptr_A,
                const J* __restrict csc_row_ind_A,
                const T* __restrict csc_val_A,
                const T* __restrict B,
                int64_t        ldb,
                rocgraph_order order_B,
                T              beta,
                T* __restrict C,
                int64_t             ldc,
                rocgraph_order      order_C,
                rocgraph_index_base base)
{
    switch(transA)
    {
    case rocgraph_operation_none:
    {
        return host_csrmm(K,
                          N,
                          M,
                          rocgraph_operation_transpose,
                          transB,
                          alpha,
                          csc_col_ptr_A,
                          csc_row_ind_A,
                          csc_val_A,
                          B,
                          ldb,
                          order_B,
                          beta,
                          C,
                          ldc,
                          order_C,
                          base,
                          false);
    }
    case rocgraph_operation_transpose:
    {
        return host_csrmm(K,
                          N,
                          M,
                          rocgraph_operation_none,
                          transB,
                          alpha,
                          csc_col_ptr_A,
                          csc_row_ind_A,
                          csc_val_A,
                          B,
                          ldb,
                          order_B,
                          beta,
                          C,
                          ldc,
                          order_C,
                          base,
                          false);
    }
    case rocgraph_operation_conjugate_transpose:
    {
        return host_csrmm(K,
                          N,
                          M,
                          rocgraph_operation_none,
                          transB,
                          alpha,
                          csc_col_ptr_A,
                          csc_row_ind_A,
                          csc_val_A,
                          B,
                          ldb,
                          order_B,
                          beta,
                          C,
                          ldc,
                          order_C,
                          base,
                          true);
    }
    }
}

template <typename T, typename I, typename J>
void host_cscmm_batched(J                   M,
                        J                   N,
                        J                   K,
                        J                   batch_count_A,
                        int64_t             offsets_batch_stride_A,
                        int64_t             rows_values_batch_stride_A,
                        rocgraph_operation  transA,
                        rocgraph_operation  transB,
                        T                   alpha,
                        const I*            csc_col_ptr_A,
                        const J*            csc_row_ind_A,
                        const T*            csc_val_A,
                        const T*            B,
                        int64_t             ldb,
                        J                   batch_count_B,
                        int64_t             batch_stride_B,
                        rocgraph_order      order_B,
                        T                   beta,
                        T*                  C,
                        int64_t             ldc,
                        J                   batch_count_C,
                        int64_t             batch_stride_C,
                        rocgraph_order      order_C,
                        rocgraph_index_base base)
{
    switch(transA)
    {
    case rocgraph_operation_none:
    {
        return host_csrmm_batched(K,
                                  N,
                                  M,
                                  batch_count_A,
                                  offsets_batch_stride_A,
                                  rows_values_batch_stride_A,
                                  rocgraph_operation_transpose,
                                  transB,
                                  alpha,
                                  csc_col_ptr_A,
                                  csc_row_ind_A,
                                  csc_val_A,
                                  B,
                                  ldb,
                                  batch_count_B,
                                  batch_stride_B,
                                  order_B,
                                  beta,
                                  C,
                                  ldc,
                                  batch_count_C,
                                  batch_stride_C,
                                  order_C,
                                  base,
                                  false);
    }
    case rocgraph_operation_transpose:
    {
        return host_csrmm_batched(K,
                                  N,
                                  M,
                                  batch_count_A,
                                  offsets_batch_stride_A,
                                  rows_values_batch_stride_A,
                                  rocgraph_operation_none,
                                  transB,
                                  alpha,
                                  csc_col_ptr_A,
                                  csc_row_ind_A,
                                  csc_val_A,
                                  B,
                                  ldb,
                                  batch_count_B,
                                  batch_stride_B,
                                  order_B,
                                  beta,
                                  C,
                                  ldc,
                                  batch_count_C,
                                  batch_stride_C,
                                  order_C,
                                  base,
                                  false);
    }
    case rocgraph_operation_conjugate_transpose:
    {
        return host_csrmm_batched(K,
                                  N,
                                  M,
                                  batch_count_A,
                                  offsets_batch_stride_A,
                                  rows_values_batch_stride_A,
                                  rocgraph_operation_none,
                                  transB,
                                  alpha,
                                  csc_col_ptr_A,
                                  csc_row_ind_A,
                                  csc_val_A,
                                  B,
                                  ldb,
                                  batch_count_B,
                                  batch_stride_B,
                                  order_B,
                                  beta,
                                  C,
                                  ldc,
                                  batch_count_C,
                                  batch_stride_C,
                                  order_C,
                                  base,
                                  true);
    }
    }
}

template <typename I, typename J, typename T>
static inline void host_lssolve(J                   M,
                                J                   nrhs,
                                rocgraph_operation  transB,
                                T                   alpha,
                                const I*            csr_row_ptr,
                                const J*            csr_col_ind,
                                const T*            csr_val,
                                T*                  B,
                                int64_t             ldb,
                                rocgraph_order      order_B,
                                rocgraph_diag_type  diag_type,
                                rocgraph_index_base base,
                                J*                  struct_pivot,
                                J*                  numeric_pivot)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(J i = 0; i < nrhs; ++i)
    {
        // Process lower triangular part
        for(J row = 0; row < M; ++row)
        {
            int64_t idx_B = (transB == rocgraph_operation_none && order_B == rocgraph_order_column)
                                ? i * ldb + row
                                : row * ldb + i;

            T sum = static_cast<T>(0);
            if(transB == rocgraph_operation_conjugate_transpose)
            {
                sum = alpha * rocgraph_conj(B[idx_B]);
            }
            else
            {
                sum = alpha * B[idx_B];
            }

            I diag      = -1;
            I row_begin = csr_row_ptr[row] - base;
            I row_end   = csr_row_ptr[row + 1] - base;

            T diag_val = static_cast<T>(0);

            for(I j = row_begin; j < row_end; ++j)
            {
                J local_col = csr_col_ind[j] - base;
                T local_val = csr_val[j];

                if(local_val == static_cast<T>(0) && local_col == row
                   && diag_type == rocgraph_diag_type_non_unit)
                {
                    // Numerical zero pivot found, avoid division by 0 and store
                    // index for later use
                    *numeric_pivot = std::min(*numeric_pivot, row + base);
                    local_val      = static_cast<T>(1);
                }

                // Ignore all entries that are above the diagonal
                if(local_col > row)
                {
                    break;
                }

                // Diagonal entry
                if(local_col == row)
                {
                    // If diagonal type is non unit, do division by diagonal entry
                    // This is not required for unit diagonal for obvious reasons
                    if(diag_type == rocgraph_diag_type_non_unit)
                    {
                        diag     = j;
                        diag_val = static_cast<T>(1) / local_val;
                    }

                    break;
                }

                // Lower triangular part
                int64_t idx
                    = (transB == rocgraph_operation_none && order_B == rocgraph_order_column)
                          ? i * ldb + local_col
                          : local_col * ldb + i;

                sum = std::fma(-local_val, B[idx], sum);
            }

            if(diag_type == rocgraph_diag_type_non_unit)
            {
                if(diag == -1)
                {
                    *struct_pivot = std::min(*struct_pivot, row + base);
                }

                B[idx_B] = sum * diag_val;
            }
            else
            {
                B[idx_B] = sum;
            }
        }
    }
}

template <typename I, typename J, typename T>
static inline void host_ussolve(J                   M,
                                J                   nrhs,
                                rocgraph_operation  transB,
                                T                   alpha,
                                const I*            csr_row_ptr,
                                const J*            csr_col_ind,
                                const T*            csr_val,
                                T*                  B,
                                int64_t             ldb,
                                rocgraph_order      order_B,
                                rocgraph_diag_type  diag_type,
                                rocgraph_index_base base,
                                J*                  struct_pivot,
                                J*                  numeric_pivot)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(J i = 0; i < nrhs; ++i)
    {
        // Process upper triangular part
        for(J row = M - 1; row >= 0; --row)
        {
            int64_t idx_B = (transB == rocgraph_operation_none && order_B == rocgraph_order_column)
                                ? i * ldb + row
                                : row * ldb + i;

            T sum = static_cast<T>(0);
            if(transB == rocgraph_operation_conjugate_transpose)
            {
                sum = alpha * rocgraph_conj(B[idx_B]);
            }
            else
            {
                sum = alpha * B[idx_B];
            }

            I diag      = -1;
            I row_begin = csr_row_ptr[row] - base;
            I row_end   = csr_row_ptr[row + 1] - base;

            T diag_val = static_cast<T>(0);

            for(I j = row_end - 1; j >= row_begin; --j)
            {
                J local_col = csr_col_ind[j] - base;
                T local_val = csr_val[j];

                // Ignore all entries that are below the diagonal
                if(local_col < row)
                {
                    continue;
                }

                // Diagonal entry
                if(local_col == row)
                {
                    if(diag_type == rocgraph_diag_type_non_unit)
                    {
                        // Check for numerical zero
                        if(local_val == static_cast<T>(0))
                        {
                            *numeric_pivot = std::min(*numeric_pivot, row + base);
                            local_val      = static_cast<T>(1);
                        }

                        diag     = j;
                        diag_val = static_cast<T>(1) / local_val;
                    }

                    continue;
                }

                // Upper triangular part
                int64_t idx
                    = (transB == rocgraph_operation_none && order_B == rocgraph_order_column)
                          ? i * ldb + local_col
                          : local_col * ldb + i;

                sum = std::fma(-local_val, B[idx], sum);
            }

            if(diag_type == rocgraph_diag_type_non_unit)
            {
                if(diag == -1)
                {
                    *struct_pivot = std::min(*struct_pivot, row + base);
                }

                B[idx_B] = sum * diag_val;
            }
            else
            {
                B[idx_B] = sum;
            }
        }
    }
}

template <typename I, typename J, typename T>
void host_csrsm(J                   M,
                J                   nrhs,
                I                   nnz,
                rocgraph_operation  transA,
                rocgraph_operation  transB,
                T                   alpha,
                const I*            csr_row_ptr,
                const J*            csr_col_ind,
                const T*            csr_val,
                T*                  B,
                int64_t             ldb,
                rocgraph_order      order_B,
                rocgraph_diag_type  diag_type,
                rocgraph_fill_mode  fill_mode,
                rocgraph_index_base base,
                J*                  struct_pivot,
                J*                  numeric_pivot)
{
    if(nrhs == 0)
    {
        *struct_pivot  = M + 1;
        *numeric_pivot = M + 1;
        *numeric_pivot = std::min(*numeric_pivot, *struct_pivot);
        *struct_pivot  = (*struct_pivot == M + 1) ? -1 : *struct_pivot;
        *numeric_pivot = (*numeric_pivot == M + 1) ? -1 : *numeric_pivot;
    }
    else if(nrhs == 1)
    {
        J B_m = (transB == rocgraph_operation_none) ? M : nrhs;
        J B_n = (transB == rocgraph_operation_none) ? nrhs : M;

        int64_t nrowB = (order_B == rocgraph_order_column) ? ldb : B_m;
        int64_t ncolB = (order_B == rocgraph_order_column) ? B_n : ldb;

        host_dense_vector<T> x(nrowB * ncolB, 0);
        host_dense_vector<T> y(M, 0);
        for(size_t i = 0; i < x.size(); i++)
        {
            x[i] = B[i];
        }

        for(J i = 0; i < M; i++)
        {
            y[i] = B[i];
        }

        int64_t x_inc
            = (transB == rocgraph_operation_none && order_B == rocgraph_order_column) ? 1 : ldb;

        host_csrsv(transA,
                   M,
                   nnz,
                   alpha,
                   csr_row_ptr,
                   csr_col_ind,
                   csr_val,
                   x.data(),
                   x_inc,
                   y.data(),
                   diag_type,
                   fill_mode,
                   base,
                   struct_pivot,
                   numeric_pivot);

        if((transB == rocgraph_operation_none && order_B == rocgraph_order_column))
        {
            for(J i = 0; i < M; i++)
            {
                B[i] = y[i];
            }
        }
        else
        {
            for(J i = 0; i < M; i++)
            {
                B[i * ldb] = y[i];
            }
        }
    }
    else
    {
        // Initialize pivot
        *struct_pivot  = M + 1;
        *numeric_pivot = M + 1;

        if(transA == rocgraph_operation_none)
        {
            if(fill_mode == rocgraph_fill_mode_lower)
            {
                host_lssolve(M,
                             nrhs,
                             transB,
                             alpha,
                             csr_row_ptr,
                             csr_col_ind,
                             csr_val,
                             B,
                             ldb,
                             order_B,
                             diag_type,
                             base,
                             struct_pivot,
                             numeric_pivot);
            }
            else
            {
                host_ussolve(M,
                             nrhs,
                             transB,
                             alpha,
                             csr_row_ptr,
                             csr_col_ind,
                             csr_val,
                             B,
                             ldb,
                             order_B,
                             diag_type,
                             base,
                             struct_pivot,
                             numeric_pivot);
            }
        }
        else if(transA == rocgraph_operation_transpose
                || transA == rocgraph_operation_conjugate_transpose)
        {
            // Transpose matrix
            host_dense_vector<I> csrt_row_ptr(M + 1);
            host_dense_vector<J> csrt_col_ind(nnz);
            host_dense_vector<T> csrt_val(nnz);

            host_csr_to_csc<I, J, T>(M,
                                     M,
                                     nnz,
                                     csr_row_ptr,
                                     csr_col_ind,
                                     csr_val,
                                     csrt_col_ind,
                                     csrt_row_ptr,
                                     csrt_val,
                                     rocgraph_action_numeric,
                                     base);

            if(transA == rocgraph_operation_conjugate_transpose)
            {
                for(size_t i = 0; i < csrt_val.size(); i++)
                {
                    csrt_val[i] = rocgraph_conj(csrt_val[i]);
                }
            }

            if(fill_mode == rocgraph_fill_mode_lower)
            {
                host_ussolve(M,
                             nrhs,
                             transB,
                             alpha,
                             csrt_row_ptr.data(),
                             csrt_col_ind.data(),
                             csrt_val.data(),
                             B,
                             ldb,
                             order_B,
                             diag_type,
                             base,
                             struct_pivot,
                             numeric_pivot);
            }
            else
            {
                host_lssolve(M,
                             nrhs,
                             transB,
                             alpha,
                             csrt_row_ptr.data(),
                             csrt_col_ind.data(),
                             csrt_val.data(),
                             B,
                             ldb,
                             order_B,
                             diag_type,
                             base,
                             struct_pivot,
                             numeric_pivot);
            }
        }

        *numeric_pivot = std::min(*numeric_pivot, *struct_pivot);

        *struct_pivot  = (*struct_pivot == M + 1) ? -1 : *struct_pivot;
        *numeric_pivot = (*numeric_pivot == M + 1) ? -1 : *numeric_pivot;
    }
}

template <typename I, typename T>
void host_coosm(I                   M,
                I                   nrhs,
                int64_t             nnz,
                rocgraph_operation  transA,
                rocgraph_operation  transB,
                T                   alpha,
                const I*            coo_row_ind,
                const I*            coo_col_ind,
                const T*            coo_val,
                T*                  B,
                int64_t             ldb,
                rocgraph_order      order_B,
                rocgraph_diag_type  diag_type,
                rocgraph_fill_mode  fill_mode,
                rocgraph_index_base base,
                I*                  struct_pivot,
                I*                  numeric_pivot)
{
    if(std::is_same<I, int32_t>() && nnz < std::numeric_limits<int32_t>::max())
    {
        host_dense_vector<int32_t> csr_row_ptr(M + 1);

        host_coo_to_csr<int32_t, I>(M, nnz, coo_row_ind, csr_row_ptr.data(), base);

        host_csrsm<int32_t, I>(M,
                               nrhs,
                               nnz,
                               transA,
                               transB,
                               alpha,
                               csr_row_ptr.data(),
                               coo_col_ind,
                               coo_val,
                               B,
                               ldb,
                               order_B,
                               diag_type,
                               fill_mode,
                               base,
                               struct_pivot,
                               numeric_pivot);
    }
    else
    {
        host_dense_vector<int64_t> csr_row_ptr(M + 1);

        host_coo_to_csr(M, nnz, coo_row_ind, csr_row_ptr.data(), base);

        host_csrsm(M,
                   nrhs,
                   nnz,
                   transA,
                   transB,
                   alpha,
                   csr_row_ptr.data(),
                   coo_col_ind,
                   coo_val,
                   B,
                   ldb,
                   order_B,
                   diag_type,
                   fill_mode,
                   base,
                   struct_pivot,
                   numeric_pivot);
    }
}

template <typename I, typename T>
void host_gemvi(I                   M,
                I                   N,
                T                   alpha,
                const T*            A,
                int64_t             lda,
                I                   nnz,
                const T*            x_val,
                const I*            x_ind,
                T                   beta,
                T*                  y,
                rocgraph_index_base base)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(I i = 0; i < M; ++i)
    {
        T sum = static_cast<T>(0);

        for(I j = 0; j < nnz; ++j)
        {
            sum = std::fma(x_val[j], A[(x_ind[j] - base) * lda + i], sum);
        }

        y[i] = std::fma(alpha, sum, beta * y[i]);
    }
}

template <typename T>
void host_gemmi(rocgraph_int        M,
                rocgraph_int        N,
                rocgraph_operation  transA,
                rocgraph_operation  transB,
                T                   alpha,
                const T*            A,
                int64_t             lda,
                const rocgraph_int* csr_row_ptr,
                const rocgraph_int* csr_col_ind,
                const T*            csr_val,
                T                   beta,
                T*                  C,
                int64_t             ldc,
                rocgraph_index_base base)
{
    if(transB == rocgraph_operation_transpose)
    {
        for(rocgraph_int i = 0; i < M; ++i)
        {
            for(rocgraph_int j = 0; j < N; ++j)
            {
                T sum = static_cast<T>(0);

                rocgraph_int row_begin = csr_row_ptr[j] - base;
                rocgraph_int row_end   = csr_row_ptr[j + 1] - base;

                for(rocgraph_int k = row_begin; k < row_end; ++k)
                {
                    rocgraph_int col_B = csr_col_ind[k] - base;
                    T            val_B = csr_val[k];
                    T            val_A = A[col_B * lda + i];

                    sum = std::fma(val_A, val_B, sum);
                }

                C[j * ldc + i] = std::fma(beta, C[j * ldc + i], alpha * sum);
            }
        }
    }
}

/*
 * ===========================================================================
 *    extra GRAPH
 * ===========================================================================
 */

/*
 * ===========================================================================
 *    conversion GRAPH
 * ===========================================================================
 */
template <typename T>
rocgraph_status host_nnz(rocgraph_direction dirA,
                         rocgraph_int       m,
                         rocgraph_int       n,
                         const T*           A,
                         int64_t            lda,
                         rocgraph_int*      nnz_per_row_columns,
                         rocgraph_int*      nnz_total_dev_host_ptr)
{

    rocgraph_int mn = (dirA == rocgraph_direction_row) ? m : n;
    for(rocgraph_int j = 0; j < mn; ++j)
    {
        nnz_per_row_columns[j] = 0;
    }

    for(rocgraph_int j = 0; j < n; ++j)
    {
        for(rocgraph_int i = 0; i < m; ++i)
        {
            if(A[j * lda + i] != 0)
            {
                if(dirA == rocgraph_direction_row)
                {
                    nnz_per_row_columns[i] += 1;
                }
                else
                {
                    nnz_per_row_columns[j] += 1;
                }
            }
        }
    }

    nnz_total_dev_host_ptr[0] = 0;
    for(rocgraph_int j = 0; j < mn; ++j)
    {
        nnz_total_dev_host_ptr[0] += nnz_per_row_columns[j];
    }

    return rocgraph_status_success;
}

template <typename T>
void host_prune_dense2csr(rocgraph_int                     m,
                          rocgraph_int                     n,
                          const host_dense_vector<T>&      A,
                          int64_t                          lda,
                          rocgraph_index_base              base,
                          T                                threshold,
                          rocgraph_int&                    nnz,
                          host_dense_vector<T>&            csr_val,
                          host_dense_vector<rocgraph_int>& csr_row_ptr,
                          host_dense_vector<rocgraph_int>& csr_col_ind)
{
    csr_row_ptr.resize(m + 1, 0);
    csr_row_ptr[0] = base;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocgraph_int i = 0; i < m; i++)
    {
        for(rocgraph_int j = 0; j < n; j++)
        {
            if(std::abs(A[lda * j + i]) > threshold)
            {
                csr_row_ptr[i + 1]++;
            }
        }
    }

    for(rocgraph_int i = 1; i <= m; i++)
    {
        csr_row_ptr[i] += csr_row_ptr[i - 1];
    }

    nnz = csr_row_ptr[m] - csr_row_ptr[0];

    csr_col_ind.resize(nnz);
    csr_val.resize(nnz);

    rocgraph_int index = 0;
    for(rocgraph_int i = 0; i < m; i++)
    {
        for(rocgraph_int j = 0; j < n; j++)
        {
            if(std::abs(A[lda * j + i]) > threshold)
            {
                csr_val[index]     = A[lda * j + i];
                csr_col_ind[index] = j + base;

                index++;
            }
        }
    }
}

template <typename T>
void host_prune_dense2csr_by_percentage(rocgraph_int                     m,
                                        rocgraph_int                     n,
                                        const host_dense_vector<T>&      A,
                                        int64_t                          lda,
                                        rocgraph_index_base              base,
                                        T                                percentage,
                                        rocgraph_int&                    nnz,
                                        host_dense_vector<T>&            csr_val,
                                        host_dense_vector<rocgraph_int>& csr_row_ptr,
                                        host_dense_vector<rocgraph_int>& csr_col_ind)
{
    rocgraph_int nnz_A = m * n;
    rocgraph_int pos   = std::ceil(nnz_A * (percentage / 100)) - 1;
    pos                = std::min(pos, nnz_A - 1);
    pos                = std::max(pos, static_cast<rocgraph_int>(0));

    host_dense_vector<T> sorted_A(m * n);
    for(rocgraph_int i = 0; i < n; i++)
    {
        for(rocgraph_int j = 0; j < m; j++)
        {
            sorted_A[m * i + j] = std::abs(A[lda * i + j]);
        }
    }

    std::sort(sorted_A.begin(), sorted_A.end());

    T threshold = sorted_A[pos];
    host_prune_dense2csr<T>(m, n, A, lda, base, threshold, nnz, csr_val, csr_row_ptr, csr_col_ind);
}

template <rocgraph_direction DIRA, typename T, typename I, typename J>
void host_dense2csx(J                   m,
                    J                   n,
                    rocgraph_index_base base,
                    const T*            A,
                    int64_t             ld,
                    rocgraph_order      order,
                    const I*            nnz_per_row_columns,
                    T*                  csx_val,
                    I*                  csx_row_col_ptr,
                    J*                  csx_col_row_ind)
{
    static constexpr T s_zero = {};
    J                  len    = (rocgraph_direction_row == DIRA) ? m : n;
    *csx_row_col_ptr          = base;
    for(J i = 0; i < len; ++i)
    {
        csx_row_col_ptr[i + 1] = nnz_per_row_columns[i] + csx_row_col_ptr[i];
    }

    switch(DIRA)
    {
    case rocgraph_direction_column:
    {
        for(J j = 0; j < n; ++j)
        {
            for(J i = 0; i < m; ++i)
            {
                if(order == rocgraph_order_column)
                {
                    if(A[j * ld + i] != s_zero)
                    {
                        *csx_val++         = A[j * ld + i];
                        *csx_col_row_ind++ = i + base;
                    }
                }
                else
                {
                    if(A[i * ld + j] != s_zero)
                    {
                        *csx_val++         = A[i * ld + j];
                        *csx_col_row_ind++ = i + base;
                    }
                }
            }
        }

        break;
    }

    case rocgraph_direction_row:
    {
        //
        // Does not matter having an orthogonal traversal ... testing only.
        // Otherwise, we would use csx_row_ptr_A to store the shifts.
        // and once the job is done a simple memory move would reinitialize the csx_row_ptr_A to its initial state)
        //
        for(J i = 0; i < m; ++i)
        {
            for(J j = 0; j < n; ++j)
            {
                if(order == rocgraph_order_column)
                {
                    if(A[j * ld + i] != s_zero)
                    {
                        *csx_val++         = A[j * ld + i];
                        *csx_col_row_ind++ = j + base;
                    }
                }
                else
                {
                    if(A[i * ld + j] != s_zero)
                    {
                        *csx_val++         = A[i * ld + j];
                        *csx_col_row_ind++ = j + base;
                    }
                }
            }
        }

        break;
    }
    }
}

template <rocgraph_direction DIRA, typename T, typename I, typename J>
void host_csx2dense(J                   m,
                    J                   n,
                    rocgraph_index_base base,
                    rocgraph_order      order,
                    const T*            csx_val,
                    const I*            csx_row_col_ptr,
                    const J*            csx_col_row_ind,
                    T*                  A,
                    int64_t             ld)
{
    if(order == rocgraph_order_column)
    {
        for(J col = 0; col < n; ++col)
        {
            for(J row = 0; row < m; ++row)
            {
                A[row + ld * col] = static_cast<T>(0);
            }
        }
    }
    else
    {
        for(J row = 0; row < m; ++row)
        {
            for(J col = 0; col < n; ++col)
            {
                A[col + ld * row] = static_cast<T>(0);
            }
        }
    }

    if(DIRA == rocgraph_direction_column)
    {
        for(J col = 0; col < n; ++col)
        {
            I start = csx_row_col_ptr[col] - base;
            I end   = csx_row_col_ptr[col + 1] - base;

            if(order == rocgraph_order_column)
            {
                for(I at = start; at < end; ++at)
                {
                    A[(csx_col_row_ind[at] - base) + ld * col] = csx_val[at];
                }
            }
            else
            {
                for(I at = start; at < end; ++at)
                {
                    A[col + ld * (csx_col_row_ind[at] - base)] = csx_val[at];
                }
            }
        }
    }
    else
    {
        for(J row = 0; row < m; ++row)
        {
            I start = csx_row_col_ptr[row] - base;
            I end   = csx_row_col_ptr[row + 1] - base;

            if(order == rocgraph_order_column)
            {
                for(I at = start; at < end; ++at)
                {
                    A[(csx_col_row_ind[at] - base) * ld + row] = csx_val[at];
                }
            }
            else
            {
                for(I at = start; at < end; ++at)
                {

                    A[row * ld + (csx_col_row_ind[at] - base)] = csx_val[at];
                }
            }
        }
    }
}

template <typename I, typename T>
void host_dense_to_coo(I                           m,
                       I                           n,
                       rocgraph_index_base         base,
                       const host_dense_vector<T>& A,
                       int64_t                     ld,
                       rocgraph_order              order,
                       const host_dense_vector<I>& nnz_per_row,
                       host_dense_vector<T>&       coo_val,
                       host_dense_vector<I>&       coo_row_ind,
                       host_dense_vector<I>&       coo_col_ind)
{
    // Find number of non-zeros in dense matrix
    int64_t nnz = 0;
    for(I i = 0; i < m; ++i)
    {
        nnz += nnz_per_row[i];
    }

    coo_val.resize(nnz, static_cast<T>(0));
    coo_row_ind.resize(nnz, 0);
    coo_col_ind.resize(nnz, 0);

    // Fill COO matrix
    int64_t index = 0;
    for(I i = 0; i < m; i++)
    {
        for(I j = 0; j < n; j++)
        {
            if(order == rocgraph_order_column)
            {
                if(A[ld * j + i] != static_cast<T>(0))
                {
                    coo_val[index]     = A[ld * j + i];
                    coo_row_ind[index] = i + base;
                    coo_col_ind[index] = j + base;

                    index++;
                }
            }
            else
            {
                if(A[ld * i + j] != static_cast<T>(0))
                {
                    coo_val[index]     = A[ld * i + j];
                    coo_row_ind[index] = i + base;
                    coo_col_ind[index] = j + base;

                    index++;
                }
            }
        }
    }
}

template <typename I, typename T>
void host_coo_to_dense(I                           m,
                       I                           n,
                       int64_t                     nnz,
                       rocgraph_index_base         base,
                       const host_dense_vector<T>& coo_val,
                       const host_dense_vector<I>& coo_row_ind,
                       const host_dense_vector<I>& coo_col_ind,
                       host_dense_vector<T>&       A,
                       int64_t                     ld,
                       rocgraph_order              order)
{
    I nm = order == rocgraph_order_column ? n : m;

    A.resize(ld * nm);

    if(order == rocgraph_order_column)
    {
        for(I i = 0; i < n; i++)
        {
            for(I j = 0; j < m; j++)
            {
                A[ld * i + j] = static_cast<T>(0);
            }
        }
    }
    else
    {
        for(I j = 0; j < m; j++)
        {
            for(I i = 0; i < n; i++)
            {
                A[ld * j + i] = static_cast<T>(0);
            }
        }
    }

    for(int64_t i = 0; i < nnz; i++)
    {
        I row = coo_row_ind[i] - base;
        I col = coo_col_ind[i] - base;
        T val = coo_val[i];

        if(order == rocgraph_order_column)
        {
            A[ld * col + row] = val;
        }
        else
        {
            A[ld * row + col] = val;
        }
    }
}

template <typename I, typename J, typename T>
void host_csr_to_csc(J                     M,
                     J                     N,
                     I                     nnz,
                     const I*              csr_row_ptr,
                     const J*              csr_col_ind,
                     const T*              csr_val,
                     host_dense_vector<J>& csc_row_ind,
                     host_dense_vector<I>& csc_col_ptr,
                     host_dense_vector<T>& csc_val,
                     rocgraph_action       action,
                     rocgraph_index_base   base)
{
    // Determine nnz per column
    for(I i = 0; i < nnz; ++i)
    {
        ++csc_col_ptr[csr_col_ind[i] + 1 - base];
    }

    // Scan
    for(J i = 0; i < N; ++i)
    {
        csc_col_ptr[i + 1] += csc_col_ptr[i];
    }

    // Fill row indices and values
    for(J i = 0; i < M; ++i)
    {
        I row_begin = csr_row_ptr[i] - base;
        I row_end   = csr_row_ptr[i + 1] - base;

        for(I j = row_begin; j < row_end; ++j)
        {
            J col = csr_col_ind[j] - base;
            I idx = csc_col_ptr[col];

            csc_row_ind[idx] = i + base;
            csc_val[idx]     = csr_val[j];

            ++csc_col_ptr[col];
        }
    }

    // Shift column pointer array
    for(J i = N; i > 0; --i)
    {
        csc_col_ptr[i] = csc_col_ptr[i - 1] + base;
    }

    if(csc_col_ptr.size() > 0)
    {
        csc_col_ptr[0] = base;
    }
}

template <typename T>
void host_coosort_by_column(rocgraph_int                     M,
                            rocgraph_int                     nnz,
                            host_dense_vector<rocgraph_int>& coo_row_ind,
                            host_dense_vector<rocgraph_int>& coo_col_ind,
                            host_dense_vector<T>&            coo_val)
{
    // Permutation vector
    host_dense_vector<rocgraph_int> perm(nnz);

    for(rocgraph_int i = 0; i < nnz; ++i)
    {
        perm[i] = i;
    }

    host_dense_vector<rocgraph_int> tmp_row(nnz);
    host_dense_vector<rocgraph_int> tmp_col(nnz);
    host_dense_vector<T>            tmp_val(nnz);

    tmp_row = coo_row_ind;
    tmp_col = coo_col_ind;
    tmp_val = coo_val;

    // Sort
    std::sort(perm.begin(), perm.end(), [&](const rocgraph_int& a, const rocgraph_int& b) {
        if(tmp_col[a] < tmp_col[b])
        {
            return true;
        }
        else if(tmp_col[a] == tmp_col[b])
        {
            return (tmp_row[a] < tmp_row[b]);
        }
        else
        {
            return false;
        }
    });

    for(rocgraph_int i = 0; i < nnz; ++i)
    {
        coo_row_ind[i] = tmp_row[perm[i]];
        coo_col_ind[i] = tmp_col[perm[i]];
        coo_val[i]     = tmp_val[perm[i]];
    }
}

// INSTANTIATE

template struct rocgraph_host<float, int32_t, int32_t>;
template struct rocgraph_host<double, int32_t, int32_t>;

template struct rocgraph_host<float, int64_t, int32_t>;
template struct rocgraph_host<double, int64_t, int32_t>;

template struct rocgraph_host<float, int64_t, int64_t>;
template struct rocgraph_host<double, int64_t, int64_t>;

#define INSTANTIATE_GATHER_SCATTER(ITYPE, TTYPE)                                                \
    template void host_gthr<ITYPE, TTYPE>(                                                      \
        ITYPE nnz, const TTYPE* y, TTYPE* x_val, const ITYPE* x_ind, rocgraph_index_base base); \
    template void host_sctr<ITYPE, TTYPE>(                                                      \
        ITYPE nnz, const TTYPE* x_val, const ITYPE* x_ind, TTYPE* y, rocgraph_index_base base)

#define INSTANTIATE_T(TYPE)                                                                              \
    template void            host_gthrz<TYPE>(rocgraph_int nnz,                                          \
                                   TYPE * y,                                                  \
                                   TYPE * x_val,                                              \
                                   const rocgraph_int* x_ind,                                 \
                                   rocgraph_index_base base);                                 \
    template rocgraph_status host_nnz<TYPE>(rocgraph_direction dirA,                                     \
                                            rocgraph_int       m,                                        \
                                            rocgraph_int       n,                                        \
                                            const TYPE*        A,                                        \
                                            int64_t            lda,                                      \
                                            rocgraph_int*      nnz_per_row_columns,                      \
                                            rocgraph_int*      nnz_total_dev_host_ptr);                       \
    template void            host_coosort_by_column<TYPE>(rocgraph_int M,                                \
                                               rocgraph_int nnz,                              \
                                               host_dense_vector<rocgraph_int> & coo_row_ind, \
                                               host_dense_vector<rocgraph_int> & coo_col_ind, \
                                               host_dense_vector<TYPE> & coo_val);

#define INSTANTIATE_IT(ITYPE, TTYPE)                                          \
    template void host_coomm<TTYPE, ITYPE>(ITYPE               M,             \
                                           ITYPE               N,             \
                                           ITYPE               K,             \
                                           int64_t             NNZ,           \
                                           rocgraph_operation  transA,        \
                                           rocgraph_operation  transB,        \
                                           TTYPE               alpha,         \
                                           const ITYPE*        coo_row_ind_A, \
                                           const ITYPE*        coo_col_ind_A, \
                                           const TTYPE*        coo_val_A,     \
                                           const TTYPE*        B,             \
                                           int64_t             ldb,           \
                                           rocgraph_order      order_B,       \
                                           TTYPE               beta,          \
                                           TTYPE*              C,             \
                                           int64_t             ldc,           \
                                           rocgraph_order      order_C,       \
                                           rocgraph_index_base base);

#define INSTANTIATE_IJT(ITYPE, JTYPE, TTYPE)                                                  \
    template void host_csr_to_csc<ITYPE, JTYPE, TTYPE>(JTYPE                     M,           \
                                                       JTYPE                     N,           \
                                                       ITYPE                     nnz,         \
                                                       const ITYPE*              csr_row_ptr, \
                                                       const JTYPE*              csr_col_ind, \
                                                       const TTYPE*              csr_val,     \
                                                       host_dense_vector<JTYPE>& csc_row_ind, \
                                                       host_dense_vector<ITYPE>& csc_col_ptr, \
                                                       host_dense_vector<TTYPE>& csc_val,     \
                                                       rocgraph_action           action,      \
                                                       rocgraph_index_base       base);             \
    template void host_csrmm<TTYPE, ITYPE, JTYPE>(JTYPE               M,                      \
                                                  JTYPE               N,                      \
                                                  JTYPE               K,                      \
                                                  rocgraph_operation  transA,                 \
                                                  rocgraph_operation  transB,                 \
                                                  TTYPE               alpha,                  \
                                                  const ITYPE*        csr_row_ptr_A,          \
                                                  const JTYPE*        csr_col_ind_A,          \
                                                  const TTYPE*        csr_val_A,              \
                                                  const TTYPE*        B,                      \
                                                  int64_t             ldb,                    \
                                                  rocgraph_order      order_B,                \
                                                  TTYPE               beta,                   \
                                                  TTYPE*              C,                      \
                                                  int64_t             ldc,                    \
                                                  rocgraph_order      order_C,                \
                                                  rocgraph_index_base base,                   \
                                                  bool                force_conj_A);                         \
    template void host_cscmm<TTYPE, ITYPE, JTYPE>(JTYPE               M,                      \
                                                  JTYPE               N,                      \
                                                  JTYPE               K,                      \
                                                  rocgraph_operation  transA,                 \
                                                  rocgraph_operation  transB,                 \
                                                  TTYPE               alpha,                  \
                                                  const ITYPE*        csc_col_ptr_A,          \
                                                  const JTYPE*        csc_row_ind_A,          \
                                                  const TTYPE*        csc_val_A,              \
                                                  const TTYPE*        B,                      \
                                                  int64_t             ldb,                    \
                                                  rocgraph_order      order_B,                \
                                                  TTYPE               beta,                   \
                                                  TTYPE*              C,                      \
                                                  int64_t             ldc,                    \
                                                  rocgraph_order      order_C,                \
                                                  rocgraph_index_base base);

#define INSTANTIATE_IJAXYT(ITYPE, JTYPE, ATYPE, XTYPE, YTYPE, TTYPE) \
    template void host_cscmv(rocgraph_operation   trans,             \
                             JTYPE                M,                 \
                             JTYPE                N,                 \
                             ITYPE                nnz,               \
                             TTYPE                alpha,             \
                             const ITYPE*         csc_col_ptr,       \
                             const JTYPE*         csc_row_ind,       \
                             const ATYPE*         csc_val,           \
                             const XTYPE*         x,                 \
                             TTYPE                beta,              \
                             YTYPE*               y,                 \
                             rocgraph_index_base  base,              \
                             rocgraph_matrix_type matrix_type,       \
                             rocgraph_spmv_alg    algo);                \
    template void host_csrmv(rocgraph_operation   trans,             \
                             JTYPE                M,                 \
                             JTYPE                N,                 \
                             ITYPE                nnz,               \
                             TTYPE                alpha,             \
                             const ITYPE*         csr_row_ptr,       \
                             const JTYPE*         csr_col_ind,       \
                             const ATYPE*         csr_val,           \
                             const XTYPE*         x,                 \
                             TTYPE                beta,              \
                             YTYPE*               y,                 \
                             rocgraph_index_base  base,              \
                             rocgraph_matrix_type matrix_type,       \
                             rocgraph_spmv_alg    algo,              \
                             bool                 force_conj)

#define INSTANTIATE_IAXYT(ITYPE, ATYPE, XTYPE, YTYPE, TTYPE)  \
    template void host_coomv(rocgraph_operation  trans,       \
                             ITYPE               M,           \
                             ITYPE               N,           \
                             int64_t             nnz,         \
                             TTYPE               alpha,       \
                             const ITYPE*        coo_row_ind, \
                             const ITYPE*        coo_col_ind, \
                             const ATYPE*        coo_val,     \
                             const XTYPE*        x,           \
                             TTYPE               beta,        \
                             YTYPE*              y,           \
                             rocgraph_index_base base)

INSTANTIATE_GATHER_SCATTER(int32_t, int8_t);
INSTANTIATE_GATHER_SCATTER(int32_t, float);
INSTANTIATE_GATHER_SCATTER(int32_t, double);
INSTANTIATE_GATHER_SCATTER(int64_t, int8_t);
INSTANTIATE_GATHER_SCATTER(int64_t, float);
INSTANTIATE_GATHER_SCATTER(int64_t, double);

INSTANTIATE_T(float);
INSTANTIATE_T(double);

INSTANTIATE_IT(int32_t, float);
INSTANTIATE_IT(int32_t, double);
INSTANTIATE_IT(int64_t, float);
INSTANTIATE_IT(int64_t, double);

INSTANTIATE_IJT(int32_t, int32_t, float);
INSTANTIATE_IJT(int32_t, int32_t, double);
INSTANTIATE_IJT(int64_t, int32_t, float);
INSTANTIATE_IJT(int64_t, int32_t, double);
INSTANTIATE_IJT(int64_t, int64_t, float);
INSTANTIATE_IJT(int64_t, int64_t, double);

INSTANTIATE_IJAXYT(int32_t, int32_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE_IJAXYT(int64_t, int32_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE_IJAXYT(int64_t, int64_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE_IJAXYT(int32_t, int32_t, int8_t, int8_t, float, float);
INSTANTIATE_IJAXYT(int64_t, int32_t, int8_t, int8_t, float, float);
INSTANTIATE_IJAXYT(int64_t, int64_t, int8_t, int8_t, float, float);

INSTANTIATE_IJAXYT(int32_t, int32_t, float, double, double, double);
INSTANTIATE_IJAXYT(int64_t, int32_t, float, double, double, double);
INSTANTIATE_IJAXYT(int64_t, int64_t, float, double, double, double);

INSTANTIATE_IJAXYT(int32_t, int32_t, float, float, float, float);
INSTANTIATE_IJAXYT(int64_t, int32_t, float, float, float, float);
INSTANTIATE_IJAXYT(int64_t, int64_t, float, float, float, float);
INSTANTIATE_IJAXYT(int32_t, int32_t, double, double, double, double);
INSTANTIATE_IJAXYT(int64_t, int32_t, double, double, double, double);
INSTANTIATE_IJAXYT(int64_t, int64_t, double, double, double, double);

INSTANTIATE_IAXYT(int32_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE_IAXYT(int64_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE_IAXYT(int32_t, int8_t, int8_t, float, float);
INSTANTIATE_IAXYT(int64_t, int8_t, int8_t, float, float);
INSTANTIATE_IAXYT(int32_t, float, double, double, double);
INSTANTIATE_IAXYT(int64_t, float, double, double, double);
INSTANTIATE_IAXYT(int32_t, float, float, float, float);
INSTANTIATE_IAXYT(int64_t, float, float, float, float);
INSTANTIATE_IAXYT(int32_t, double, double, double, double);
INSTANTIATE_IAXYT(int64_t, double, double, double, double);
