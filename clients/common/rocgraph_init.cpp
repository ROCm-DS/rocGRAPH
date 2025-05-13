/*! \file */

// Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_init.hpp"
#include "rocgraph_import.hpp"
#include "rocgraph_importer_impls.hpp"
#include "rocgraph_matrix.hpp"

template <typename I, typename J>
void host_coo_to_csr(J M, I nnz, const J* coo_row_ind, I* csr_row_ptr, rocgraph_index_base base)
{
    // Resize and initialize csr_row_ptr with zeros
    for(size_t i = 0; i < M + 1; ++i)
    {
        csr_row_ptr[i] = 0;
    }

    for(size_t i = 0; i < nnz; ++i)
    {
        ++csr_row_ptr[coo_row_ind[i] + 1 - base];
    }

    csr_row_ptr[0] = base;
    for(J i = 0; i < M; ++i)
    {
        csr_row_ptr[i + 1] += csr_row_ptr[i];
    }
}

template <typename I, typename J>
void host_csr_to_coo(J                           M,
                     I                           nnz,
                     const host_dense_vector<I>& csr_row_ptr,
                     host_dense_vector<J>&       coo_row_ind,
                     rocgraph_index_base         base)
{
    // Resize coo_row_ind
    coo_row_ind.resize(nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(J i = 0; i < M; ++i)
    {
        I row_begin = csr_row_ptr[i] - base;
        I row_end   = csr_row_ptr[i + 1] - base;

        for(I j = row_begin; j < row_end; ++j)
        {
            coo_row_ind[j] = i + base;
        }
    }
}

template <typename I, typename J>
void host_csr_to_coo_aos(J                           M,
                         I                           nnz,
                         const host_dense_vector<I>& csr_row_ptr,
                         const host_dense_vector<J>& csr_col_ind,
                         host_dense_vector<I>&       coo_ind,
                         rocgraph_index_base         base)
{
    // Resize coo_ind
    coo_ind.resize(2 * nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(I i = 0; i < M; ++i)
    {
        I row_begin = csr_row_ptr[i] - base;
        I row_end   = csr_row_ptr[i + 1] - base;

        for(I j = row_begin; j < row_end; ++j)
        {
            coo_ind[2 * j]     = i + base;
            coo_ind[2 * j + 1] = static_cast<I>(csr_col_ind[j]);
        }
    }
}

/* ==================================================================================== */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);

// Initialize vector with random values

template <typename T>
void rocgraph_init_exact(
    T* A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count, int a, int b)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
    {
        for(size_t j = 0; j < N; ++j)
        {
            for(size_t i = 0; i < M; ++i)
            {
                A[i + j * lda + i_batch * stride] = random_cached_generator_exact<T>(a, b);
            }
        }
    }
}

template <typename T>
void rocgraph_init(
    T* A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count, T a, T b)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
    {
        for(size_t j = 0; j < N; ++j)
        {
            for(size_t i = 0; i < M; ++i)
            {
                A[i + j * lda + i_batch * stride] = random_cached_generator<T>(a, b);
            }
        }
    }
}

template <typename T>
void rocgraph_init_exact(host_dense_vector<T>& A,
                         size_t                M,
                         size_t                N,
                         size_t                lda,
                         size_t                stride,
                         size_t                batch_count,
                         int                   a,
                         int                   b)
{
    rocgraph_init_exact(A.data(), M, N, lda, stride, batch_count, a, b);
}

template <typename T>
void rocgraph_init(host_dense_vector<T>& A,
                   size_t                M,
                   size_t                N,
                   size_t                lda,
                   size_t                stride,
                   size_t                batch_count,
                   T                     a,
                   T                     b)
{
    rocgraph_init(A.data(), M, N, lda, stride, batch_count, a, b);
}

// Initializes graph index vector with nnz entries ranging from start to end
template <typename I>
void rocgraph_init_index(host_dense_vector<I>& x, size_t nnz, size_t start, size_t end)
{
    host_dense_vector<int> check(end - start, 0);

    size_t num = 0;

    while(num < nnz)
    {
        I val = random_generator_exact<I>(start, end - 1);
        if(!check[val - start])
        {
            x[num++]           = val;
            check[val - start] = 1;
        }
    }

    std::sort(x.data(), ((I*)x.data()) + x.size());
}

// Initialize matrix so adjacent entries have alternating sign.
// In gemm if either A or B are initialized with alernating
// sign the reduction sum will be summing positive
// and negative numbers, so it should not get too large.
// This helps reduce floating point inaccuracies for 16bit
// arithmetic where the exponent has only 5 bits, and the
// mantissa 10 bits.
template <typename T>
void rocgraph_init_alternating_sign(
    host_dense_vector<T>& A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
            {
                auto value                        = random_cached_generator_exact<T>();
                A[i + j * lda + i_batch * stride] = (i ^ j) & 1 ? value : -value;
            }
}

/* ==================================================================================== */
/*! \brief  Initialize an array with random data, with NaN where appropriate */

template <typename T>
void rocgraph_init_nan(T* A, size_t N)
{
    for(size_t i = 0; i < N; ++i)
        A[i] = T(rocgraph_nan_rng());
}

template <typename T>
void rocgraph_init_nan(
    host_dense_vector<T>& A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(rocgraph_nan_rng());
}

/* ==================================================================================== */
/*! \brief  Generate a random graph matrix in COO format */
template <typename I, typename T>
void rocgraph_init_coo_matrix(host_dense_vector<I>& row_ind,
                              host_dense_vector<I>& col_ind,
                              host_dense_vector<T>& val,
                              I                     M,
                              I                     N,
                              int64_t               nnz,
                              rocgraph_index_base   base,
                              bool                  full_rank,
                              bool                  to_int)
{
    if(nnz == 0)
    {
        row_ind.resize(nnz);
        col_ind.resize(nnz);
        val.resize(nnz);
        return;
    }
    // If M > N, full rank is not possible
    if(full_rank && M > N)
    {
        std::cerr << "ERROR: M > N, cannot generate matrix with full rank" << std::endl;
        full_rank = false;
    }

    // If nnz < M, full rank is not possible
    if(full_rank && nnz < M)
    {
        std::cerr << "ERROR: nnz < M, cannot generate matrix with full rank" << std::endl;
        full_rank = false;
    }

    if(row_ind.size() != nnz)
    {
        row_ind.resize(nnz);
    }
    if(col_ind.size() != nnz)
    {
        col_ind.resize(nnz);
    }
    if(val.size() != nnz)
    {
        val.resize(nnz);
    }

    // Generate histogram of non-zero counts per row based on average non-zeros per row
    host_dense_vector<I> count(M, 0);
    I                    start = full_rank ? (I)std::min((int64_t)M, nnz) : 0;
    if(full_rank)
    {
        for(I k = 0; k < start; ++k)
        {
            count[k] = 1;
        }
    }

    int64_t remaining_nnz   = nnz - start;
    I       avg_nnz_per_row = remaining_nnz / M;

    for(I k = 0; k < M; k++)
    {
        I nnz_in_row = std::min(random_cached_generator_exact<I>(0, 2 * avg_nnz_per_row), N);
        nnz_in_row   = (I)std::min(remaining_nnz, (int64_t)nnz_in_row);

        count[k] += nnz_in_row;

        remaining_nnz -= nnz_in_row;
    }

    // Sprinkle any remaining non-zeros amoung the rows
    for(int64_t k = 0; k < remaining_nnz; ++k)
    {
        I   i       = random_generator_exact<I>(0, M - 1);
        int maxiter = 0;
        while(count[i] >= N && maxiter++ < 10)
        {
            i = random_generator_exact<I>(0, M - 1);
        }
        if(maxiter >= 10)
        {
            for(i = 0; i < M; ++i)
            {
                if(count[i] < N)
                {
                    break;
                }
            }
            if(i == M)
            {
                std::cerr << "rocgraph_init_coo_matrix error" << std::endl;
                exit(1);
            }
        }
        count[i] += 1;
    }

    // Compute row index array from non-zeros per row count histogram
    int64_t offset          = 0;
    I       max_nnz_per_row = count[0];
    for(I k = 0; k < M; k++)
    {
        I nnz_in_row = count[k];
        if(max_nnz_per_row < nnz_in_row)
            max_nnz_per_row = nnz_in_row;

        for(I i = 0; i < nnz_in_row; i++)
        {
            row_ind[offset + i] = k;
        }

        offset += nnz_in_row;
    }

    // Generate column index array with values clustered around the diagonal
    I                    sec = std::min(2 * max_nnz_per_row, N);
    host_dense_vector<I> random(2 * sec + 1);
    int64_t              at = 0;
    for(I i = 0; i < M; ++i)
    {
        int64_t begin      = at;
        I       nnz_in_row = count[i];
        I       bmax       = std::min(i + sec, N - 1);
        I       bmin       = std::max(bmax - 2 * sec, ((I)0));

        // Initial permutation of column indices
        for(I k = 0; k <= (bmax - bmin); ++k)
        {
            random[k] = k;
        }

        // shuffle permutation
        for(I k = 0; k < nnz_in_row; ++k)
        {
            std::swap(random[k], random[random_generator_exact<I>(0, bmax - bmin)]);
        }

        if(full_rank)
        {
            col_ind[at++] = i;
            for(I k = 1; k < nnz_in_row; ++k)
            {
                if(bmin + random[k] == i)
                {
                    col_ind[at++] = bmin + random[bmax - bmin];
                }
                else
                {
                    col_ind[at++] = bmin + random[k];
                }
            }
        }
        else
        {
            for(I k = 0; k < nnz_in_row; ++k)
            {
                col_ind[at++] = bmin + random[k];
            }
        }

        if(nnz_in_row > 0)
        {
            std::sort(col_ind.data() + begin, col_ind.data() + begin + nnz_in_row);
        }
    }

    // Correct index base accordingly
    if(base == rocgraph_index_base_one)
    {
        for(int64_t i = 0; i < nnz; ++i)
        {
            ++row_ind[i];
            ++col_ind[i];
        }
    }

    if(to_int)
    {
        // Sample random values
        for(int64_t i = 0; i < nnz; ++i)
        {
            val[i] = random_cached_generator_exact<T>();
        }
    }
    else
    {
        if(full_rank)
        {
            for(int64_t i = 0; i < nnz; ++i)
            {
                if(row_ind[i] == col_ind[i])
                {
                    // Sample diagonal values
                    val[i] = random_cached_generator<T>(static_cast<T>(4.0), static_cast<T>(8.0));
                    val[i] += val[i]
                              * random_cached_generator<T>(static_cast<T>(-1.0e-2),
                                                           static_cast<T>(1.0e-2));
                }
                else
                {
                    // Samples off-diagonal values
                    val[i] = random_cached_generator<T>(static_cast<T>(-0.5), static_cast<T>(0.5));
                }
            }
        }
        else
        {
            for(int64_t i = 0; i < nnz; ++i)
            {
                val[i] = random_cached_generator<T>(static_cast<T>(-1.0), static_cast<T>(1.0));
            }
        }
    }
}

/* ==================================================================================== */
/*! \brief  Generate 2D 9pt laplacian on unit square in CSR format */
template <typename I, typename J, typename T>
void rocgraph_init_csr_laplace2d(host_dense_vector<I>& row_ptr,
                                 host_dense_vector<J>& col_ind,
                                 host_dense_vector<T>& val,
                                 int32_t               dim_x,
                                 int32_t               dim_y,
                                 J&                    M,
                                 J&                    N,
                                 I&                    nnz,
                                 rocgraph_index_base   base)
{
    // Do nothing
    if(dim_x == 0 || dim_y == 0)
    {
        return;
    }

    M = dim_x * dim_y;
    N = dim_x * dim_y;

    // Approximate 9pt stencil
    I nnz_mat = 9 * M;

    row_ptr.resize(M + 1);
    col_ind.resize(nnz_mat);
    val.resize(nnz_mat);

    nnz        = base;
    row_ptr[0] = base;

    // Fill local arrays
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int32_t iy = 0; iy < dim_y; ++iy)
    {
        for(int32_t ix = 0; ix < dim_x; ++ix)
        {
            J row = iy * dim_x + ix;

            for(int32_t sy = -1; sy <= 1; ++sy)
            {
                if(iy + sy > -1 && iy + sy < dim_y)
                {
                    for(int32_t sx = -1; sx <= 1; ++sx)
                    {
                        if(ix + sx > -1 && ix + sx < dim_x)
                        {
                            J col = row + sy * dim_x + sx;

                            col_ind[nnz - base] = col + base;
                            val[nnz - base]     = (col == row) ? 8.0 : -1.0;

                            ++nnz;
                        }
                    }
                }
            }

            row_ptr[row + 1] = nnz;
        }
    }

    // Adjust nnz by index base
    nnz -= base;

    // compress to actual nnz
    col_ind.resize(nnz);
    val.resize(nnz);
}

/* ==================================================================================== */
/*! \brief  Generate 2D 9pt laplacian on unit square in COO format */
template <typename I, typename T>
void rocgraph_init_coo_laplace2d(host_dense_vector<I>& row_ind,
                                 host_dense_vector<I>& col_ind,
                                 host_dense_vector<T>& val,
                                 int32_t               dim_x,
                                 int32_t               dim_y,
                                 I&                    M,
                                 I&                    N,
                                 int64_t&              nnz,
                                 rocgraph_index_base   base)
{
    // Always load using int64 as we dont know ahead of time how many nnz exist in matrix
    host_dense_vector<int64_t> row_ptr;

    // Sample CSR matrix
    rocgraph_init_csr_laplace2d(row_ptr, col_ind, val, dim_x, dim_y, M, N, nnz, base);

    // Convert to COO
    host_csr_to_coo(M, nnz, row_ptr, row_ind, base);
}

/* ==================================================================================== */
/*! \brief  Generate 3D 27pt laplacian on unit square in CSR format */
template <typename I, typename J, typename T>
void rocgraph_init_csr_laplace3d(host_dense_vector<I>& row_ptr,
                                 host_dense_vector<J>& col_ind,
                                 host_dense_vector<T>& val,
                                 int32_t               dim_x,
                                 int32_t               dim_y,
                                 int32_t               dim_z,
                                 J&                    M,
                                 J&                    N,
                                 I&                    nnz,
                                 rocgraph_index_base   base)
{
    // Do nothing
    if(dim_x == 0 || dim_y == 0 || dim_z == 0)
    {
        return;
    }

    M = dim_x * dim_y * dim_z;
    N = dim_x * dim_y * dim_z;

    // Approximate 27pt stencil
    I nnz_mat = 27 * M;

    row_ptr.resize(M + 1);
    col_ind.resize(nnz_mat);
    val.resize(nnz_mat);

    nnz        = base;
    row_ptr[0] = base;

    // Fill local arrays
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int32_t iz = 0; iz < dim_z; ++iz)
    {
        for(int32_t iy = 0; iy < dim_y; ++iy)
        {
            for(int32_t ix = 0; ix < dim_x; ++ix)
            {
                J row = iz * dim_x * dim_y + iy * dim_x + ix;

                for(int32_t sz = -1; sz <= 1; ++sz)
                {
                    if(iz + sz > -1 && iz + sz < dim_z)
                    {
                        for(int32_t sy = -1; sy <= 1; ++sy)
                        {
                            if(iy + sy > -1 && iy + sy < dim_y)
                            {
                                for(int32_t sx = -1; sx <= 1; ++sx)
                                {
                                    if(ix + sx > -1 && ix + sx < dim_x)
                                    {
                                        J col = row + sz * dim_x * dim_y + sy * dim_x + sx;

                                        col_ind[nnz - base] = col + base;
                                        val[nnz - base]     = (col == row) ? 26.0 : -1.0;

                                        ++nnz;
                                    }
                                }
                            }
                        }
                    }
                }

                row_ptr[row + 1] = nnz;
            }
        }
    }

    // Adjust nnz by index base
    nnz -= base;

    // compress to actual nnz
    col_ind.resize(nnz);
    val.resize(nnz);
}

/* ==================================================================================== */
/*! \brief  Generate 3D 27pt laplacian on unit square in COO format */
template <typename I, typename T>
void rocgraph_init_coo_laplace3d(host_dense_vector<I>& row_ind,
                                 host_dense_vector<I>& col_ind,
                                 host_dense_vector<T>& val,
                                 int32_t               dim_x,
                                 int32_t               dim_y,
                                 int32_t               dim_z,
                                 I&                    M,
                                 I&                    N,
                                 int64_t&              nnz,
                                 rocgraph_index_base   base)
{
    // Always load using int64 as we dont know ahead of time how many nnz exist in matrix
    host_dense_vector<int64_t> row_ptr;

    // Sample CSR matrix
    rocgraph_init_csr_laplace3d(row_ptr, col_ind, val, dim_x, dim_y, dim_z, M, N, nnz, base);

    // Convert to COO
    host_csr_to_coo(M, nnz, row_ptr, row_ind, base);
}

/* ==================================================================================== */
/*! \brief  Read matrix from mtx file in CSR format */
template <typename I, typename J, typename T>
void rocgraph_init_csr_mtx(const char*           filename,
                           host_dense_vector<I>& csr_row_ptr,
                           host_dense_vector<J>& csr_col_ind,
                           host_dense_vector<T>& csr_val,
                           J&                    M,
                           J&                    N,
                           I&                    nnz,
                           rocgraph_index_base   base)
{
    I       coo_M, coo_N;
    int64_t coo_nnz;

    host_dense_vector<I> coo_row_ind;
    host_dense_vector<I> coo_col_ind;

    // Read COO matrix
    rocgraph_init_coo_mtx(filename, coo_row_ind, coo_col_ind, csr_val, coo_M, coo_N, coo_nnz, base);

    // Convert to CSR
    M   = (J)coo_M;
    N   = (J)coo_N;
    nnz = (I)coo_nnz;

    csr_row_ptr.resize(M + 1);
    csr_col_ind.resize(nnz);

    host_coo_to_csr(coo_M, nnz, coo_row_ind.data(), csr_row_ptr.data(), base);

    for(I i = 0; i < nnz; ++i)
    {
        csr_col_ind[i] = (J)coo_col_ind[i];
    }
}

/* ============================================================================================ */
/*! \brief  Read matrix from mtx file in COO format */
template <typename I, typename T>
void rocgraph_init_coo_mtx(const char*           filename,
                           host_dense_vector<I>& coo_row_ind,
                           host_dense_vector<I>& coo_col_ind,
                           host_dense_vector<T>& coo_val,
                           I&                    M,
                           I&                    N,
                           int64_t&              nnz,
                           rocgraph_index_base   base)
{
    rocgraph_importer_matrixmarket importer(filename);
    rocgraph_status                status
        = rocgraph_import_graph_coo(importer, coo_row_ind, coo_col_ind, coo_val, M, N, nnz, base);
    CHECK_ROCGRAPH_THROW_ERROR(status);
}

/* ==================================================================================== */
/*! \brief  Read matrix from smtx file in CSR format */
template <typename I, typename J, typename T>
void rocgraph_init_csr_smtx(const char*           filename,
                            host_dense_vector<I>& csr_row_ptr,
                            host_dense_vector<J>& csr_col_ind,
                            host_dense_vector<T>& csr_val,
                            J&                    M,
                            J&                    N,
                            I&                    nnz,
                            rocgraph_index_base   base)
{

    rocgraph_importer_mlcsr importer(filename);
    const rocgraph_status   status
        = rocgraph_import_graph_csr(importer, csr_row_ptr, csr_col_ind, csr_val, M, N, nnz, base);

    for(size_t i = 0; i < nnz; ++i)
    {
        csr_val[i] = random_cached_generator<T>();
    }

    CHECK_ROCGRAPH_THROW_ERROR(status);
}

/* ============================================================================================ */
/*! \brief  Read matrix from smtx file in COO format */
template <typename I, typename T>
void rocgraph_init_coo_smtx(const char*           filename,
                            host_dense_vector<I>& coo_row_ind,
                            host_dense_vector<I>& coo_col_ind,
                            host_dense_vector<T>& coo_val,
                            I&                    M,
                            I&                    N,
                            int64_t&              nnz,
                            rocgraph_index_base   base)
{
    host_dense_vector<int64_t> csr_row_ptr;
    rocgraph_init_csr_smtx<int64_t, I, T>(
        filename, csr_row_ptr, coo_col_ind, coo_val, M, N, nnz, base);
    coo_row_ind.resize(nnz);
    host_csr_to_coo(M, nnz, csr_row_ptr, coo_row_ind, base);
}

template <typename I, typename J, typename T>
void rocgraph_init_csr_rocalution(const char*           filename,
                                  host_dense_vector<I>& row_ptr,
                                  host_dense_vector<J>& col_ind,
                                  host_dense_vector<T>& val,
                                  J&                    M,
                                  J&                    N,
                                  I&                    nnz,
                                  rocgraph_index_base   base)
{
    rocgraph_importer_rocalution importer(filename);
    rocgraph_status              status
        = rocgraph_import_graph_csr(importer, row_ptr, col_ind, val, M, N, nnz, base);
    CHECK_ROCGRAPH_THROW_ERROR(status);
}

/* ==================================================================================== */
/*! \brief  Read matrix from binary file in rocALUTION format */
template <typename I, typename T>
void rocgraph_init_coo_rocalution(const char*           filename,
                                  host_dense_vector<I>& row_ind,
                                  host_dense_vector<I>& col_ind,
                                  host_dense_vector<T>& val,
                                  I&                    M,
                                  I&                    N,
                                  int64_t&              nnz,
                                  rocgraph_index_base   base)
{
    I                    csr_nnz = 0;
    host_dense_vector<I> row_ptr(M + 1);

    // Sample CSR matrix
    rocgraph_init_csr_rocalution(filename, row_ptr, col_ind, val, M, N, csr_nnz, base);

    host_csr_to_coo(M, csr_nnz, row_ptr, row_ind, base);
    nnz = csr_nnz;
}

/* ==================================================================================== */
/*! \brief  Generate a random graph matrix in CSR format */
template <typename I, typename J, typename T>
void rocgraph_init_csr_random(host_dense_vector<I>&     csr_row_ptr,
                              host_dense_vector<J>&     csr_col_ind,
                              host_dense_vector<T>&     csr_val,
                              J                         M,
                              J                         N,
                              I&                        nnz,
                              rocgraph_index_base       base,
                              rocgraph_matrix_init_kind init_kind,
                              bool                      full_rank,
                              bool                      to_int)
{
    switch(init_kind)
    {
    case rocgraph_matrix_init_kind_tunedavg:
    {
        rocgraph_int alpha = static_cast<rocgraph_int>(0);
        if(N >= 16384)
        {
            alpha = static_cast<rocgraph_int>(8);
        }
        else if(N >= 8192)
        {
            alpha = static_cast<rocgraph_int>(8);
        }
        else if(N >= 4096)
        {
            alpha = static_cast<rocgraph_int>(16);
        }
        else if(N >= 1024)
        {
            alpha = static_cast<rocgraph_int>(32);
        }
        else
        {
            alpha = static_cast<rocgraph_int>(64);
        }

        nnz = static_cast<I>(M) * alpha;
        nnz = std::min(nnz, static_cast<I>(M) * static_cast<I>(N));

        // Sample random matrix
        host_dense_vector<J> row_ind(nnz);
        // Sample COO matrix
        rocgraph_init_coo_matrix<J>(
            row_ind, csr_col_ind, csr_val, M, N, nnz, base, full_rank, to_int);

        // Convert to CSR
        csr_row_ptr.resize(M + 1);
        host_coo_to_csr(M, nnz, row_ind.data(), csr_row_ptr.data(), base);
        break;
    }

    case rocgraph_matrix_init_kind_default:
    {
        if(M < 32 && N < 32)
        {
            nnz = (static_cast<I>(M) * static_cast<I>(N)) / 4;
            if(full_rank)
            {
                nnz = std::max(nnz, static_cast<I>(M));
            }
            nnz = std::max(nnz, static_cast<I>(M));
            nnz = std::min(nnz, static_cast<I>(M) * static_cast<I>(N));
        }
        else
        {
            nnz = static_cast<I>(M) * ((M > 1000 || N > 1000) ? 2.0 / std::max(M, N) : 0.02)
                  * static_cast<I>(N);
        }

        // Sample random matrix
        host_dense_vector<J> row_ind(nnz);
        // Sample COO matrix
        rocgraph_init_coo_matrix<J>(
            row_ind, csr_col_ind, csr_val, M, N, nnz, base, full_rank, to_int);

        // Convert to CSR
        csr_row_ptr.resize(M + 1);
        host_coo_to_csr(M, nnz, row_ind.data(), csr_row_ptr.data(), base);
        break;
    }
    }
}

/* ==================================================================================== */
/*! \brief  Generate a random graph matrix in COO format */
template <typename I, typename T>
void rocgraph_init_coo_random(host_dense_vector<I>&     row_ind,
                              host_dense_vector<I>&     col_ind,
                              host_dense_vector<T>&     val,
                              I                         M,
                              I                         N,
                              int64_t&                  nnz,
                              rocgraph_index_base       base,
                              rocgraph_matrix_init_kind init_kind,
                              bool                      full_rank,
                              bool                      to_int)
{
    switch(init_kind)
    {
    case rocgraph_matrix_init_kind_tunedavg:
    {
        rocgraph_int alpha = static_cast<rocgraph_int>(0);
        if(N >= 16384)
        {
            alpha = static_cast<rocgraph_int>(8);
        }
        else if(N >= 8192)
        {
            alpha = static_cast<rocgraph_int>(16);
        }
        else if(N >= 4096)
        {
            alpha = static_cast<rocgraph_int>(32);
        }
        else if(N >= 1024)
        {
            alpha = static_cast<rocgraph_int>(64);
        }
        else
        {
            alpha = static_cast<rocgraph_int>(128);
        }

        nnz = static_cast<int64_t>(M) * alpha;
        nnz = std::min(nnz, static_cast<int64_t>(M) * static_cast<int64_t>(N));

        // Sample random matrix
        rocgraph_init_coo_matrix(row_ind, col_ind, val, M, N, nnz, base, full_rank, to_int);
        break;
    }
    case rocgraph_matrix_init_kind_default:
    {
        // Compute non-zero entries of the matrix
        if(M < 32 && N < 32)
        {
            nnz = (static_cast<int64_t>(M) * static_cast<int64_t>(N)) / 4;
        }
        else
        {
            nnz = static_cast<int64_t>(M) * ((M > 1000 || N > 1000) ? 2.0 / std::max(M, N) : 0.02)
                  * static_cast<int64_t>(N);
        }

        // Sample random matrix
        rocgraph_init_coo_matrix(row_ind, col_ind, val, M, N, nnz, base, full_rank, to_int);
        break;
    }
    }
}

/* ==================================================================================== */
/*! \brief  Generate a tridiagonal graph matrix in COO format */
template <typename I, typename T>
void rocgraph_init_coo_tridiagonal(host_dense_vector<I>& row_ind,
                                   host_dense_vector<I>& col_ind,
                                   host_dense_vector<T>& val,
                                   I                     M,
                                   I                     N,
                                   int64_t&              nnz,
                                   rocgraph_index_base   base,
                                   I                     l,
                                   I                     u)
{
    if(l >= 0 || -l >= M)
    {
        std::cerr << "ERROR: l >= 0 || -l >= M" << std::endl;
        return;
    }

    if(u <= 0 || u >= N)
    {
        std::cerr << "ERROR: u <= 0 || u >= N" << std::endl;
        return;
    }

    int64_t l_length = std::min((M + l), N);
    int64_t d_length = std::min(M, N);
    int64_t u_length = std::min((N - u), M);

    nnz = l_length + d_length + u_length;

    row_ind.resize(nnz);
    col_ind.resize(nnz);
    val.resize(nnz);

    int64_t index = 0;
    for(I i = 0; i < M; i++)
    {
        I l_col = i + l;
        I d_col = i;
        I u_col = i + u;

        if(l_col >= 0 && l_col < N)
        {
            row_ind[index] = i + base;
            col_ind[index] = l_col + base;
            val[index]     = random_cached_generator<T>(static_cast<T>(-1.0), static_cast<T>(1.0));
            index++;
        }

        if(d_col >= 0 && d_col < N)
        {
            row_ind[index] = i + base;
            col_ind[index] = d_col + base;
            val[index]     = random_cached_generator<T>(static_cast<T>(2.0), static_cast<T>(4.0));
            index++;
        }

        if(u_col >= 0 && u_col < N)
        {
            row_ind[index] = i + base;
            col_ind[index] = u_col + base;
            val[index]     = random_cached_generator<T>(static_cast<T>(-1.0), static_cast<T>(1.0));
            index++;
        }
    }
}

/* ==================================================================================== */
/*! \brief  Generate a tridiagonal graph matrix in CSR format */
template <typename I, typename J, typename T>
void rocgraph_init_csr_tridiagonal(host_dense_vector<I>& row_ptr,
                                   host_dense_vector<J>& col_ind,
                                   host_dense_vector<T>& val,
                                   J                     M,
                                   J                     N,
                                   I&                    nnz,
                                   rocgraph_index_base   base,
                                   J                     l,
                                   J                     u)
{
    int64_t              coo_nnz;
    host_dense_vector<J> row_ind;
    // Sample COO matrix
    rocgraph_init_coo_tridiagonal<J>(row_ind, col_ind, val, M, N, coo_nnz, base, l, u);

    if(std::is_same<I, int32_t>() && coo_nnz > std::numeric_limits<int32_t>::max())
    {
        std::cerr << "Error: Attempting to create CSR tridiagonal matrix with more than "
                  << std::numeric_limits<int32_t>::max()
                  << " non-zeros while using int32_t row indexing." << std::endl;
        exit(1);
    }

    nnz = (I)coo_nnz;

    // Convert to CSR
    row_ptr.resize(M + 1);
    host_coo_to_csr(M, nnz, row_ind.data(), row_ptr.data(), base);
}

/* ==================================================================================== */
/*! \brief  Generate a pentadiagonal graph matrix in COO format */
template <typename I, typename T>
void rocgraph_init_coo_pentadiagonal(host_dense_vector<I>& row_ind,
                                     host_dense_vector<I>& col_ind,
                                     host_dense_vector<T>& val,
                                     I                     M,
                                     I                     N,
                                     int64_t&              nnz,
                                     rocgraph_index_base   base,
                                     I                     ll,
                                     I                     l,
                                     I                     u,
                                     I                     uu)
{
    if(ll >= 0 || l >= 0 || ll >= l || -l >= M || -ll >= M)
    {
        std::cerr << "ERROR: ll >= 0 || l >= 0 || ll >= l || -l >= M || -ll >= M" << std::endl;
        return;
    }

    if(u <= 0 || uu <= 0 || uu <= u || u >= N || uu >= N)
    {
        std::cerr << "ERROR: u <= 0 || uu <= 0 || uu <= u || u >= N || uu >= N" << std::endl;
        return;
    }

    int64_t l_length  = std::min((M + l), N);
    int64_t ll_length = std::min((M + ll), N);
    int64_t d_length  = std::min(M, N);
    int64_t u_length  = std::min((N - u), M);
    int64_t uu_length = std::min((N - uu), M);

    nnz = ll_length + l_length + d_length + u_length + uu_length;

    row_ind.resize(nnz);
    col_ind.resize(nnz);
    val.resize(nnz);

    int64_t index = 0;
    for(I i = 0; i < M; i++)
    {
        I ll_col = i + ll;
        I l_col  = i + l;
        I d_col  = i;
        I u_col  = i + u;
        I uu_col = i + uu;

        if(ll_col >= 0 && ll_col < N)
        {
            row_ind[index] = i + base;
            col_ind[index] = ll_col + base;
            val[index]     = random_cached_generator<T>(static_cast<T>(-1.0), static_cast<T>(1.0));
            index++;
        }

        if(l_col >= 0 && l_col < N)
        {
            row_ind[index] = i + base;
            col_ind[index] = l_col + base;
            val[index]     = random_cached_generator<T>(static_cast<T>(-1.0), static_cast<T>(1.0));
            index++;
        }

        if(d_col >= 0 && d_col < N)
        {
            row_ind[index] = i + base;
            col_ind[index] = d_col + base;
            val[index]     = random_cached_generator<T>(static_cast<T>(4.0), static_cast<T>(6.0));
            index++;
        }

        if(u_col >= 0 && u_col < N)
        {
            row_ind[index] = i + base;
            col_ind[index] = u_col + base;
            val[index]     = random_cached_generator<T>(static_cast<T>(-1.0), static_cast<T>(1.0));
            index++;
        }

        if(uu_col >= 0 && uu_col < N)
        {
            row_ind[index] = i + base;
            col_ind[index] = uu_col + base;
            val[index]     = random_cached_generator<T>(static_cast<T>(-1.0), static_cast<T>(1.0));
            index++;
        }
    }
}

/* ==================================================================================== */
/*! \brief  Generate a pentadiagonal graph matrix in CSR format */
template <typename I, typename J, typename T>
void rocgraph_init_csr_pentadiagonal(host_dense_vector<I>& row_ptr,
                                     host_dense_vector<J>& col_ind,
                                     host_dense_vector<T>& val,
                                     J                     M,
                                     J                     N,
                                     I&                    nnz,
                                     rocgraph_index_base   base,
                                     J                     ll,
                                     J                     l,
                                     J                     u,
                                     J                     uu)
{
    int64_t              coo_nnz;
    host_dense_vector<J> row_ind;
    // Sample COO matrix
    rocgraph_init_coo_pentadiagonal<J>(row_ind, col_ind, val, M, N, coo_nnz, base, ll, l, u, uu);

    if(std::is_same<I, int32_t>() && coo_nnz > std::numeric_limits<int32_t>::max())
    {
        std::cerr << "Error: Attempting to create CSR pentadiagonal matrix with more than "
                  << std::numeric_limits<int32_t>::max()
                  << " non-zeros while using int32_t row indexing." << std::endl;
        exit(1);
    }

    nnz = (I)coo_nnz;

    // Convert to CSR
    row_ptr.resize(M + 1);
    host_coo_to_csr(M, nnz, row_ind.data(), row_ptr.data(), base);
}

#define INSTANTIATEI(TYPE)                   \
    template void rocgraph_init_index<TYPE>( \
        host_dense_vector<TYPE> & x, size_t nnz, size_t start, size_t end);

#define INSTANTIATE(TYPE)                                                           \
    template void rocgraph_init<TYPE>(TYPE * A,                                     \
                                      size_t M,                                     \
                                      size_t N,                                     \
                                      size_t lda,                                   \
                                      size_t stride,                                \
                                      size_t batch_count = 1,                       \
                                      TYPE   a           = static_cast<TYPE>(0),    \
                                      TYPE   b           = static_cast<TYPE>(1));               \
    template void rocgraph_init_exact<TYPE>(TYPE * A,                               \
                                            size_t M,                               \
                                            size_t N,                               \
                                            size_t lda,                             \
                                            size_t stride,                          \
                                            size_t batch_count,                     \
                                            int    a = 1,                           \
                                            int    b = 10);                            \
    template void rocgraph_init<TYPE>(host_dense_vector<TYPE> & A,                  \
                                      size_t M,                                     \
                                      size_t N,                                     \
                                      size_t lda,                                   \
                                      size_t stride,                                \
                                      size_t batch_count = 1,                       \
                                      TYPE   a           = static_cast<TYPE>(0),    \
                                      TYPE   b           = static_cast<TYPE>(1));               \
    template void rocgraph_init_exact<TYPE>(host_dense_vector<TYPE> & A,            \
                                            size_t M,                               \
                                            size_t N,                               \
                                            size_t lda,                             \
                                            size_t stride,                          \
                                            size_t batch_count,                     \
                                            int    a = 1,                           \
                                            int    b = 10);                            \
    template void rocgraph_init_alternating_sign<TYPE>(host_dense_vector<TYPE> & A, \
                                                       size_t M,                    \
                                                       size_t N,                    \
                                                       size_t lda,                  \
                                                       size_t stride,               \
                                                       size_t batch_count);         \
    template void rocgraph_init_nan<TYPE>(TYPE * A, size_t N);                      \
    template void rocgraph_init_nan<TYPE>(host_dense_vector<TYPE> & A,              \
                                          size_t M,                                 \
                                          size_t N,                                 \
                                          size_t lda,                               \
                                          size_t stride = 0,                        \
                                          size_t batch_count);

#define INSTANTIATE1(ITYPE, JTYPE)                                                               \
    template void host_csr_to_coo<ITYPE, JTYPE>(JTYPE                           M,               \
                                                ITYPE                           nnz,             \
                                                const host_dense_vector<ITYPE>& csr_row_ptr,     \
                                                host_dense_vector<JTYPE>&       coo_row_ind,     \
                                                rocgraph_index_base             base);                       \
    template void host_coo_to_csr<ITYPE, JTYPE>(JTYPE               M,                           \
                                                ITYPE               NNZ,                         \
                                                const JTYPE*        coo_row_ind,                 \
                                                ITYPE*              csr_row_ptr,                 \
                                                rocgraph_index_base base);                       \
    template void host_csr_to_coo_aos<ITYPE, JTYPE>(JTYPE                           M,           \
                                                    ITYPE                           nnz,         \
                                                    const host_dense_vector<ITYPE>& csr_row_ptr, \
                                                    const host_dense_vector<JTYPE>& csr_col_ind, \
                                                    host_dense_vector<ITYPE>&       coo_ind,     \
                                                    rocgraph_index_base             base);

#define INSTANTIATE2(ITYPE, TTYPE)                                                                \
    template void rocgraph_init_coo_tridiagonal<ITYPE, TTYPE>(host_dense_vector<ITYPE> & row_ind, \
                                                              host_dense_vector<ITYPE> & col_ind, \
                                                              host_dense_vector<TTYPE> & val,     \
                                                              ITYPE M,                            \
                                                              ITYPE N,                            \
                                                              int64_t & nnz,                      \
                                                              rocgraph_index_base base,           \
                                                              ITYPE               l,              \
                                                              ITYPE               u);                           \
    template void rocgraph_init_coo_pentadiagonal<ITYPE, TTYPE>(                                  \
        host_dense_vector<ITYPE> & row_ind,                                                       \
        host_dense_vector<ITYPE> & col_ind,                                                       \
        host_dense_vector<TTYPE> & val,                                                           \
        ITYPE M,                                                                                  \
        ITYPE N,                                                                                  \
        int64_t & nnz,                                                                            \
        rocgraph_index_base base,                                                                 \
        ITYPE               ll,                                                                   \
        ITYPE               l,                                                                    \
        ITYPE               u,                                                                    \
        ITYPE               uu);                                                                                \
    template void rocgraph_init_coo_laplace2d<ITYPE, TTYPE>(host_dense_vector<ITYPE> & row_ind,   \
                                                            host_dense_vector<ITYPE> & col_ind,   \
                                                            host_dense_vector<TTYPE> & val,       \
                                                            int32_t dim_x,                        \
                                                            int32_t dim_y,                        \
                                                            ITYPE & M,                            \
                                                            ITYPE & N,                            \
                                                            int64_t & nnz,                        \
                                                            rocgraph_index_base base);            \
    template void rocgraph_init_coo_matrix<ITYPE, TTYPE>(host_dense_vector<ITYPE> & row_ind,      \
                                                         host_dense_vector<ITYPE> & col_ind,      \
                                                         host_dense_vector<TTYPE> & val,          \
                                                         ITYPE               M,                   \
                                                         ITYPE               N,                   \
                                                         int64_t             nnz,                 \
                                                         rocgraph_index_base base,                \
                                                         bool                full_rank,           \
                                                         bool                to_int);                            \
    template void rocgraph_init_coo_laplace3d<ITYPE, TTYPE>(host_dense_vector<ITYPE> & row_ind,   \
                                                            host_dense_vector<ITYPE> & col_ind,   \
                                                            host_dense_vector<TTYPE> & val,       \
                                                            int32_t dim_x,                        \
                                                            int32_t dim_y,                        \
                                                            int32_t dim_z,                        \
                                                            ITYPE & M,                            \
                                                            ITYPE & N,                            \
                                                            int64_t & nnz,                        \
                                                            rocgraph_index_base base);            \
    template void rocgraph_init_coo_mtx<ITYPE, TTYPE>(const char*               filename,         \
                                                      host_dense_vector<ITYPE>& coo_row_ind,      \
                                                      host_dense_vector<ITYPE>& coo_col_ind,      \
                                                      host_dense_vector<TTYPE>& coo_val,          \
                                                      ITYPE&                    M,                \
                                                      ITYPE&                    N,                \
                                                      int64_t&                  nnz,              \
                                                      rocgraph_index_base       base);                  \
    template void rocgraph_init_coo_smtx<ITYPE, TTYPE>(const char*               filename,        \
                                                       host_dense_vector<ITYPE>& coo_row_ind,     \
                                                       host_dense_vector<ITYPE>& coo_col_ind,     \
                                                       host_dense_vector<TTYPE>& coo_val,         \
                                                       ITYPE&                    M,               \
                                                       ITYPE&                    N,               \
                                                       int64_t&                  nnz,             \
                                                       rocgraph_index_base       base);                 \
    template void rocgraph_init_coo_rocalution<ITYPE, TTYPE>(const char*               filename,  \
                                                             host_dense_vector<ITYPE>& row_ind,   \
                                                             host_dense_vector<ITYPE>& col_ind,   \
                                                             host_dense_vector<TTYPE>& val,       \
                                                             ITYPE&                    M,         \
                                                             ITYPE&                    N,         \
                                                             int64_t&                  nnz,       \
                                                             rocgraph_index_base       base);           \
    template void rocgraph_init_coo_random<ITYPE, TTYPE>(host_dense_vector<ITYPE> & row_ind,      \
                                                         host_dense_vector<ITYPE> & col_ind,      \
                                                         host_dense_vector<TTYPE> & val,          \
                                                         ITYPE M,                                 \
                                                         ITYPE N,                                 \
                                                         int64_t & nnz,                           \
                                                         rocgraph_index_base       base,          \
                                                         rocgraph_matrix_init_kind init_kind,     \
                                                         bool                      full_rank,     \
                                                         bool                      to_int);

#define INSTANTIATE3(ITYPE, JTYPE, TTYPE)                               \
    template void rocgraph_init_csr_tridiagonal<ITYPE, JTYPE, TTYPE>(   \
        host_dense_vector<ITYPE> & row_ptr,                             \
        host_dense_vector<JTYPE> & col_ind,                             \
        host_dense_vector<TTYPE> & val,                                 \
        JTYPE M,                                                        \
        JTYPE N,                                                        \
        ITYPE & nnz,                                                    \
        rocgraph_index_base base,                                       \
        JTYPE               l,                                          \
        JTYPE               u);                                                       \
    template void rocgraph_init_csr_pentadiagonal<ITYPE, JTYPE, TTYPE>( \
        host_dense_vector<ITYPE> & row_ptr,                             \
        host_dense_vector<JTYPE> & col_ind,                             \
        host_dense_vector<TTYPE> & val,                                 \
        JTYPE M,                                                        \
        JTYPE N,                                                        \
        ITYPE & nnz,                                                    \
        rocgraph_index_base base,                                       \
        JTYPE               ll,                                         \
        JTYPE               l,                                          \
        JTYPE               u,                                          \
        JTYPE               uu);                                                      \
    template void rocgraph_init_csr_laplace2d<ITYPE, JTYPE, TTYPE>(     \
        host_dense_vector<ITYPE> & row_ptr,                             \
        host_dense_vector<JTYPE> & col_ind,                             \
        host_dense_vector<TTYPE> & val,                                 \
        int32_t dim_x,                                                  \
        int32_t dim_y,                                                  \
        JTYPE & M,                                                      \
        JTYPE & N,                                                      \
        ITYPE & nnz,                                                    \
        rocgraph_index_base base);                                      \
    template void rocgraph_init_csr_laplace3d<ITYPE, JTYPE, TTYPE>(     \
        host_dense_vector<ITYPE> & row_ptr,                             \
        host_dense_vector<JTYPE> & col_ind,                             \
        host_dense_vector<TTYPE> & val,                                 \
        int32_t dim_x,                                                  \
        int32_t dim_y,                                                  \
        int32_t dim_z,                                                  \
        JTYPE & M,                                                      \
        JTYPE & N,                                                      \
        ITYPE & nnz,                                                    \
        rocgraph_index_base base);                                      \
    template void rocgraph_init_csr_mtx<ITYPE, JTYPE, TTYPE>(           \
        const char*               filename,                             \
        host_dense_vector<ITYPE>& csr_row_ptr,                          \
        host_dense_vector<JTYPE>& csr_col_ind,                          \
        host_dense_vector<TTYPE>& csr_val,                              \
        JTYPE&                    M,                                    \
        JTYPE&                    N,                                    \
        ITYPE&                    nnz,                                  \
        rocgraph_index_base       base);                                      \
    template void rocgraph_init_csr_smtx<ITYPE, JTYPE, TTYPE>(          \
        const char*               filename,                             \
        host_dense_vector<ITYPE>& csr_row_ptr,                          \
        host_dense_vector<JTYPE>& csr_col_ind,                          \
        host_dense_vector<TTYPE>& csr_val,                              \
        JTYPE&                    M,                                    \
        JTYPE&                    N,                                    \
        ITYPE&                    nnz,                                  \
        rocgraph_index_base       base);                                      \
    template void rocgraph_init_csr_rocalution<ITYPE, JTYPE, TTYPE>(    \
        const char*               filename,                             \
        host_dense_vector<ITYPE>& row_ptr,                              \
        host_dense_vector<JTYPE>& col_ind,                              \
        host_dense_vector<TTYPE>& val,                                  \
        JTYPE&                    M,                                    \
        JTYPE&                    N,                                    \
        ITYPE&                    nnz,                                  \
        rocgraph_index_base       base);                                      \
    template void rocgraph_init_csr_random<ITYPE, JTYPE, TTYPE>(        \
        host_dense_vector<ITYPE> & row_ptr,                             \
        host_dense_vector<JTYPE> & col_ind,                             \
        host_dense_vector<TTYPE> & val,                                 \
        JTYPE M,                                                        \
        JTYPE N,                                                        \
        ITYPE & nnz,                                                    \
        rocgraph_index_base       base,                                 \
        rocgraph_matrix_init_kind init_kind,                            \
        bool                      full_rank,                            \
        bool                      to_int)

INSTANTIATEI(int32_t);
INSTANTIATEI(int64_t);

INSTANTIATE(int8_t);
INSTANTIATE(int32_t);
INSTANTIATE(int64_t);
INSTANTIATE(size_t);
INSTANTIATE(float);
INSTANTIATE(double);

INSTANTIATE1(int32_t, int32_t);
INSTANTIATE1(int64_t, int32_t);
INSTANTIATE1(int64_t, int64_t);

INSTANTIATE2(int32_t, int8_t);
INSTANTIATE2(int64_t, int8_t);
INSTANTIATE2(int32_t, float);
INSTANTIATE2(int64_t, float);
INSTANTIATE2(int32_t, double);
INSTANTIATE2(int64_t, double);

INSTANTIATE3(int32_t, int32_t, int8_t);
INSTANTIATE3(int64_t, int32_t, int8_t);
INSTANTIATE3(int64_t, int64_t, int8_t);
INSTANTIATE3(int32_t, int32_t, float);
INSTANTIATE3(int64_t, int32_t, float);
INSTANTIATE3(int64_t, int64_t, float);
INSTANTIATE3(int32_t, int32_t, double);
INSTANTIATE3(int64_t, int32_t, double);
INSTANTIATE3(int64_t, int64_t, double);
