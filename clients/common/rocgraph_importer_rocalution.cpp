/*! \file */

// Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_importer_rocalution.hpp"

static inline void read_csr_values(std::ifstream& in, int64_t nnz, int8_t* csr_val)
{
    // Temporary array to convert from double to float
    std::vector<double> tmp(nnz);

    // Read in double values
    in.read((char*)tmp.data(), sizeof(double) * nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int64_t i = 0; i < nnz; ++i)
    {
        csr_val[i] = static_cast<int8_t>(tmp[i]);
    }
}

static inline void read_csr_values(std::ifstream& in, int64_t nnz, float* csr_val)
{
    // Temporary array to convert from double to float
    std::vector<double> tmp(nnz);

    // Read in double values
    in.read((char*)tmp.data(), sizeof(double) * nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int64_t i = 0; i < nnz; ++i)
    {
        csr_val[i] = static_cast<float>(tmp[i]);
    }
}

static inline void read_csr_values(std::ifstream& in, int64_t nnz, double* csr_val)
{
    in.read((char*)csr_val, sizeof(double) * nnz);
}

rocgraph_importer_rocalution::rocgraph_importer_rocalution(const std::string& filename_)
    : m_filename(filename_)
{
}

template <typename I, typename J>
rocgraph_status rocgraph_importer_rocalution::import_graph_gebsx(rocgraph_direction* dir,
                                                                 rocgraph_direction* dirb,
                                                                 J*                  mb,
                                                                 J*                  nb,
                                                                 I*                  nnzb,
                                                                 J*                  block_dim_row,
                                                                 J* block_dim_column,
                                                                 rocgraph_index_base* base)
{
    return rocgraph_status_not_implemented;
}

template <typename T, typename I, typename J>
rocgraph_status rocgraph_importer_rocalution::import_graph_gebsx(I* ptr, J* ind, T* val)
{
    return rocgraph_status_not_implemented;
}

template <typename I>
rocgraph_status rocgraph_importer_rocalution::import_graph_coo(I*                   m,
                                                               I*                   n,
                                                               int64_t*             nnz,
                                                               rocgraph_index_base* base)
{
    return rocgraph_status_not_implemented;
}

template <typename T, typename I>
rocgraph_status rocgraph_importer_rocalution::import_graph_coo(I* row_ind, I* col_ind, T* val)
{
    return rocgraph_status_not_implemented;
}

template <typename I, typename J>
rocgraph_status rocgraph_importer_rocalution::import_graph_csx(
    rocgraph_direction* dir, J* m, J* n, I* nnz, rocgraph_index_base* base)
{

    const char* env = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        std::cout << "Opening file '" << this->m_filename << "' ... " << std::endl;
    }

    this->m_info_csx.in = new std::ifstream(this->m_filename, std::ios::in | std::ios::binary);
    if(!this->m_info_csx.in->is_open())
    {
        missing_file_error_message(this->m_filename.c_str());
        return rocgraph_status_internal_error;
    }
    std::string header;
    std::getline(this->m_info_csx.in[0], header);
    if(header != "#rocALUTION binary csr file")
    {
        return rocgraph_status_internal_error;
    }
    int version;
    this->m_info_csx.in->read((char*)&version, sizeof(int));
    int iM;
    int iN;
    int innz;
    this->m_info_csx.in->read((char*)&iM, sizeof(int));
    this->m_info_csx.in->read((char*)&iN, sizeof(int));
    this->m_info_csx.in->read((char*)&innz, sizeof(int));

    rocgraph_status status;
    status = rocgraph_type_conversion(iM, m[0]);
    if(status != rocgraph_status_success)
        return status;

    status = rocgraph_type_conversion(iN, n[0]);
    if(status != rocgraph_status_success)
        return status;

    status = rocgraph_type_conversion(innz, nnz[0]);
    if(status != rocgraph_status_success)
        return status;

    dir[0]               = rocgraph_direction_row;
    base[0]              = rocgraph_index_base_zero;
    this->m_info_csx.m   = iM;
    this->m_info_csx.nnz = innz;

    return rocgraph_status_success;
}

template <typename T, typename I, typename J>
rocgraph_status rocgraph_importer_rocalution::import_graph_csx(I* ptr, J* ind, T* val)
{
    const size_t M   = this->m_info_csx.m;
    const size_t nnz = this->m_info_csx.nnz;

    const bool same_ptr_type = std::is_same<I, int>();
    const bool same_ind_type = std::is_same<J, int>();
    const bool same_val_type = std::is_same<T, double>();
    const bool is_consistent = same_ptr_type && same_ind_type && same_val_type;
    if(is_consistent)
    {
        this->m_info_csx.in->read((char*)ptr, sizeof(int) * (M + 1));
        this->m_info_csx.in->read((char*)ind, sizeof(int) * nnz);
        this->m_info_csx.in->read((char*)val, sizeof(T) * nnz);
        this->m_info_csx.in->close();
        delete this->m_info_csx.in;
        this->m_info_csx.in = nullptr;
        {
            const char* env = getenv("GTEST_LISTENER");
            if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
            {
                std::cout << "Import done." << std::endl;
            }
        }
    }
    else
    {
        void* tmp_ptr = (void*)ptr;
        void* tmp_ind = (void*)ind;
        void* tmp_val = (void*)val;

        host_dense_vector<int> tmp_ptrv;
        host_dense_vector<int> tmp_indv;
        host_dense_vector<T>   tmp_valv;

        if(!same_ptr_type)
        {
            tmp_ptrv.resize((M + 1));
            tmp_ptr = tmp_ptrv;
        }
        else
        {
            tmp_ptr = ptr;
        }

        if(!same_ind_type)
        {
            tmp_indv.resize(nnz);
            tmp_ind = tmp_indv;
        }
        else
        {
            tmp_ind = ind;
        }

        if(!same_val_type)
        {
            tmp_valv.resize(nnz);
            tmp_val = tmp_valv;
        }
        else
        {
            tmp_val = val;
        }
        this->m_info_csx.in->read((char*)tmp_ptr, sizeof(int) * (M + 1));
        this->m_info_csx.in->read((char*)tmp_ind, sizeof(int) * nnz);
        read_csr_values(this->m_info_csx.in[0], (int64_t)nnz, (T*)tmp_val);
        //  this->m_info_csx.in->read((char*)tmp_val, sizeof(double) * nnz);
        this->m_info_csx.in->close();
        delete this->m_info_csx.in;
        this->m_info_csx.in = nullptr;
        {
            const char* env = getenv("GTEST_LISTENER");
            if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
            {
                std::cout << "Import done." << std::endl;
            }
        }
        //
        // Copy back.
        //
        if(!same_ptr_type)
        {

            rocgraph_importer_copy_mixed_arrays(M + 1, ptr, (int*)tmp_ptr);
        }
        if(!same_ind_type)
        {
            rocgraph_importer_copy_mixed_arrays(nnz, ind, (int*)tmp_ind);
        }
        if(!same_val_type)
        {
            rocgraph_importer_copy_mixed_arrays(nnz, val, (T*)tmp_val);
        }
    }

    return rocgraph_status_success;
}

#define INSTANTIATE_TIJ(T, I, J)                                                         \
    template rocgraph_status rocgraph_importer_rocalution::import_graph_csx(I*, J*, T*); \
    template rocgraph_status rocgraph_importer_rocalution::import_graph_gebsx(I*, J*, T*)

#define INSTANTIATE_TI(T, I)                                                 \
    template rocgraph_status rocgraph_importer_rocalution::import_graph_coo( \
        I* row_ind, I* col_ind, T* val)

#define INSTANTIATE_I(I)                                                     \
    template rocgraph_status rocgraph_importer_rocalution::import_graph_coo( \
        I* m, I* n, int64_t* nnz, rocgraph_index_base* base)

#define INSTANTIATE_IJ(I, J)                                                   \
    template rocgraph_status rocgraph_importer_rocalution::import_graph_csx(   \
        rocgraph_direction*, J*, J*, I*, rocgraph_index_base*);                \
    template rocgraph_status rocgraph_importer_rocalution::import_graph_gebsx( \
        rocgraph_direction*, rocgraph_direction*, J*, J*, I*, J*, J*, rocgraph_index_base*)

INSTANTIATE_I(int32_t);
INSTANTIATE_I(int64_t);

INSTANTIATE_IJ(int32_t, int32_t);
INSTANTIATE_IJ(int64_t, int32_t);
INSTANTIATE_IJ(int64_t, int64_t);

INSTANTIATE_TIJ(int8_t, int32_t, int32_t);
INSTANTIATE_TIJ(int8_t, int64_t, int32_t);
INSTANTIATE_TIJ(int8_t, int64_t, int64_t);

INSTANTIATE_TIJ(float, int32_t, int32_t);
INSTANTIATE_TIJ(float, int64_t, int32_t);
INSTANTIATE_TIJ(float, int64_t, int64_t);

INSTANTIATE_TIJ(double, int32_t, int32_t);
INSTANTIATE_TIJ(double, int64_t, int32_t);
INSTANTIATE_TIJ(double, int64_t, int64_t);

INSTANTIATE_TI(int8_t, int32_t);
INSTANTIATE_TI(int8_t, int64_t);

INSTANTIATE_TI(float, int32_t);
INSTANTIATE_TI(float, int64_t);

INSTANTIATE_TI(double, int32_t);
INSTANTIATE_TI(double, int64_t);
