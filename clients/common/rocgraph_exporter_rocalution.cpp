/*! \file */

// Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_exporter_rocalution.hpp"
template <typename X, typename Y>
rocgraph_status rocgraph_type_conversion(const X& x, Y& y);

rocgraph_exporter_rocalution::~rocgraph_exporter_rocalution()
{
    const char* env = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        std::cout << "Export done." << std::endl;
    }
}

rocgraph_exporter_rocalution::rocgraph_exporter_rocalution(const std::string& filename_)
    : m_filename(filename_)
{

    const char* env = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        std::cout << "Opening file '" << this->m_filename << "' ... " << std::endl;
    }
}

template <typename T>
rocgraph_status rocalution_write_graph_csx(
    const char* filename, int m, int n, int nnz, const int* ptr, const int* col, const T* val)
{
    std::ofstream out(filename, std::ios::out | std::ios::binary);

    if(!out.is_open())
    {
        return rocgraph_status_internal_error;
    }

    // Header
    out << "#rocALUTION binary csr file" << std::endl;

    // rocALUTION version
    int version = 10602;
    out.write((char*)&version, sizeof(int));

    // Data
    out.write((char*)&m, sizeof(int));
    out.write((char*)&n, sizeof(int));
    out.write((char*)&nnz, sizeof(int));
    out.write((char*)ptr, (m + 1) * sizeof(int));
    out.write((char*)col, nnz * sizeof(int));
    out.write((char*)val, nnz * sizeof(T));
    out.close();

    return rocgraph_status_success;
}

template <typename T>
void convert_array(int nnz, const void* data, void* mem)
{
    memcpy(mem, data, sizeof(T) * nnz);
}

template <>
void convert_array<float>(int nnz, const void* data, void* mem)
{
    double*      pmem  = (double*)mem;
    const float* pdata = (const float*)data;
    for(int i = 0; i < nnz; ++i)
    {
        pmem[i] = pdata[i];
    }
}

template <typename T, typename I, typename J>
rocgraph_status rocgraph_exporter_rocalution::write_graph_csx(rocgraph_direction dir_,
                                                              J                  m_,
                                                              J                  n_,
                                                              I                  nnz_,
                                                              const I* __restrict__ ptr_,
                                                              const J* __restrict__ ind_,
                                                              const T* __restrict__ val_,
                                                              rocgraph_index_base base_)
{

    if(dir_ != rocgraph_direction_row)
    {
        return rocgraph_status_not_implemented;
    }
    int             m;
    int             n;
    int             nnz;
    rocgraph_status status;

    status = rocgraph_type_conversion(m_, m);
    if(status != rocgraph_status_success)
    {
        return status;
    }

    status = rocgraph_type_conversion(n_, n);
    if(status != rocgraph_status_success)
    {
        return status;
    }

    status = rocgraph_type_conversion(nnz_, nnz);
    if(status != rocgraph_status_success)
    {
        return status;
    }

    const int*    ptr = nullptr;
    const int*    ind = nullptr;
    const double* val = nullptr;

    int*                  ptr_mem  = nullptr;
    int*                  ind_mem  = nullptr;
    double*               val_mem  = nullptr;
    static constexpr bool ptr_same = std::is_same<I, int>();
    static constexpr bool ind_same = std::is_same<J, int>();
    static constexpr bool val_same = std::is_same<T, double>();

    bool is_T_complex = false;

    ptr_mem = nullptr;
    if(!ptr_same && (base_ != rocgraph_index_base_zero))
    {
        rocgraph_hipHostMalloc(&ptr_mem, sizeof(int) * (m + 1));
    }

    ind_mem = nullptr;
    if(!ind_same && (base_ != rocgraph_index_base_zero))
    {
        rocgraph_hipHostMalloc(&ind_mem, sizeof(int) * nnz);
    }

    val_mem = nullptr;
    if(!val_same)
    {
        rocgraph_hipHostMalloc(&val_mem, sizeof(double) * (is_T_complex ? (2 * nnz) : nnz));
    }

    ptr = (ptr_same || (base_ == rocgraph_index_base_zero)) ? ((const int*)ptr_) : ptr_mem;
    ind = (ind_same || (base_ == rocgraph_index_base_zero)) ? ((const int*)ind_) : ind_mem;
    val = (val_same) ? ((const double*)val_) : val_mem;

    if(ptr_mem != nullptr)
    {
        for(int i = 0; i < m + 1; ++i)
        {
            status = rocgraph_type_conversion(ptr_[i], ptr_mem[i]);
            if(status != rocgraph_status_success)
            {
                break;
            }
        }

        if(base_ == rocgraph_index_base_one)
        {
            for(int i = 0; i < m + 1; ++i)
            {
                ptr_mem[i] = ptr_mem[i] - 1;
            }
        }

        if(status != rocgraph_status_success)
        {
            return status;
        }
    }

    if(ind_mem != nullptr)
    {
        for(int i = 0; i < nnz; ++i)
        {
            status = rocgraph_type_conversion(ind_[i], ind_mem[i]);
            if(status != rocgraph_status_success)
            {
                break;
            }
        }
        if(status != rocgraph_status_success)
        {
            return status;
        }
        if(base_ == rocgraph_index_base_one)
        {
            for(int i = 0; i < nnz; ++i)
            {
                ind_mem[i] = ind_mem[i] - 1;
            }
        }
    }

    if(val_mem != nullptr)
    {
        convert_array<T>(nnz, (const void*)val_, (void*)val_mem);
    }

    if(status != rocgraph_status_success)
    {
        return status;
    }

    status = rocalution_write_graph_csx(this->m_filename.c_str(), m, n, nnz, ptr, ind, val);
    if(val_mem != nullptr)
    {
        rocgraph_hipFree(val_mem);
        val_mem = nullptr;
    }
    if(ind_mem != nullptr)
    {
        rocgraph_hipFree(ind_mem);
        ind_mem = nullptr;
    }
    if(ptr_mem != nullptr)
    {
        rocgraph_hipFree(ptr_mem);
        ptr_mem = nullptr;
    }

    return status;
}

template <typename T, typename I, typename J>
rocgraph_status rocgraph_exporter_rocalution::write_graph_gebsx(rocgraph_direction dir_,
                                                                rocgraph_direction dirb_,
                                                                J                  mb_,
                                                                J                  nb_,
                                                                I                  nnzb_,
                                                                J                  block_dim_row_,
                                                                J block_dim_column_,
                                                                const I* __restrict__ ptr_,
                                                                const J* __restrict__ ind_,
                                                                const T* __restrict__ val_,
                                                                rocgraph_index_base base_)
{
    return rocgraph_status_not_implemented;
}

template <typename T, typename I>
rocgraph_status
    rocgraph_exporter_rocalution::write_dense_vector(I nmemb_, const T* __restrict__ x_, I incx_)
{
    return rocgraph_status_not_implemented;
}

template <typename T, typename I>
rocgraph_status rocgraph_exporter_rocalution::write_dense_matrix(
    rocgraph_order order_, I m_, I n_, const T* __restrict__ x_, I ld_)
{
    return rocgraph_status_not_implemented;
}

template <typename T, typename I>
rocgraph_status rocgraph_exporter_rocalution::write_graph_coo(I m_,
                                                              I n_,
                                                              I nnz_,
                                                              const I* __restrict__ row_ind_,
                                                              const I* __restrict__ col_ind_,
                                                              const T* __restrict__ val_,
                                                              rocgraph_index_base base_)
{
    return rocgraph_status_not_implemented;
}

#define INSTANTIATE_TIJ(T, I, J)                                                                  \
    template rocgraph_status rocgraph_exporter_rocalution::write_graph_csx(rocgraph_direction,    \
                                                                           J,                     \
                                                                           J,                     \
                                                                           I,                     \
                                                                           const I* __restrict__, \
                                                                           const J* __restrict__, \
                                                                           const T* __restrict__, \
                                                                           rocgraph_index_base);  \
    template rocgraph_status rocgraph_exporter_rocalution::write_graph_gebsx(                     \
        rocgraph_direction,                                                                       \
        rocgraph_direction,                                                                       \
        J,                                                                                        \
        J,                                                                                        \
        I,                                                                                        \
        J,                                                                                        \
        J,                                                                                        \
        const I* __restrict__,                                                                    \
        const J* __restrict__,                                                                    \
        const T* __restrict__,                                                                    \
        rocgraph_index_base)

#define INSTANTIATE_TI(T, I)                                                                      \
    template rocgraph_status rocgraph_exporter_rocalution::write_dense_vector(                    \
        I, const T* __restrict__, I);                                                             \
    template rocgraph_status rocgraph_exporter_rocalution::write_dense_matrix(                    \
        rocgraph_order, I, I, const T* __restrict__, I);                                          \
    template rocgraph_status rocgraph_exporter_rocalution::write_graph_coo(I,                     \
                                                                           I,                     \
                                                                           I,                     \
                                                                           const I* __restrict__, \
                                                                           const I* __restrict__, \
                                                                           const T* __restrict__, \
                                                                           rocgraph_index_base)

INSTANTIATE_TIJ(float, int32_t, int32_t);
INSTANTIATE_TIJ(float, int64_t, int32_t);
INSTANTIATE_TIJ(float, int64_t, int64_t);

INSTANTIATE_TIJ(double, int32_t, int32_t);
INSTANTIATE_TIJ(double, int64_t, int32_t);
INSTANTIATE_TIJ(double, int64_t, int64_t);

INSTANTIATE_TI(float, int32_t);
INSTANTIATE_TI(float, int64_t);

INSTANTIATE_TI(double, int32_t);
INSTANTIATE_TI(double, int64_t);
