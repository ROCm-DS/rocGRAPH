/*! \file */

// Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_importer_mlcsr.hpp"
#include <stdio.h>

rocgraph_importer_mlcsr::~rocgraph_importer_mlcsr()
{
    if(this->m_f != nullptr)
    {
        fclose(this->m_f);
    }
}

rocgraph_importer_mlcsr::rocgraph_importer_mlcsr(const std::string& filename_)
    : m_filename(filename_)
{
}

template <typename I, typename J>
rocgraph_status rocgraph_importer_mlcsr::import_graph_csx(
    rocgraph_direction* dir, J* m, J* n, I* nnz, rocgraph_index_base* base)
{
    char line[1024];
    this->m_f = fopen(this->m_filename.c_str(), "r");
    if(!this->m_f)
    {
        missing_file_error_message(this->m_filename.c_str());
        return rocgraph_status_internal_error;
    }

    //
    // Skip header.
    //
    while(0 != fgets(line, 1024, this->m_f))
    {
        const char* l = &line[0];
        while(l[0] != '\0' && (l[0] == ' ' || l[0] == '\t'))
        {
            ++l;
        }
        if(l[0] != '\0' && l[0] != '%')
        {
            break;
        }
    }

    //
    // Read dimension.
    //
    size_t inrow;
    size_t incol;
    size_t innz;
    if(EOF == sscanf(line, "%zd %zd %zd", &inrow, &incol, &innz))
    {
        return rocgraph_status_internal_error;
    }

    //
    // Convert.
    //
    rocgraph_status status;
    status = rocgraph_type_conversion(inrow, m[0]);
    if(status != rocgraph_status_success)
        return status;

    status = rocgraph_type_conversion(incol, n[0]);
    if(status != rocgraph_status_success)
        return status;

    status = rocgraph_type_conversion(innz, nnz[0]);
    if(status != rocgraph_status_success)
        return status;

    dir[0]  = rocgraph_direction_row;
    base[0] = rocgraph_index_base_zero;

    this->m_m   = inrow;
    this->m_k   = incol;
    this->m_nnz = innz;

    return rocgraph_status_success;
}

template <typename T, typename I, typename J>
rocgraph_status rocgraph_importer_mlcsr::import_graph_csx(I* ptr, J* ind, T* val)
{
    size_t          k;
    rocgraph_status status;
    for(size_t i = 0; i <= this->m_m; ++i)
    {
        if(EOF == fscanf(this->m_f, "%zd", &k))
        {
            return rocgraph_status_internal_error;
        }
        status = rocgraph_type_conversion(k, ptr[i]);
        if(status != rocgraph_status_success)
        {
            return status;
        }
    }
    for(size_t i = 0; i < this->m_nnz; ++i)
    {
        if(EOF == fscanf(this->m_f, "%zd", &k))
        {
            return rocgraph_status_internal_error;
        }
        status = rocgraph_type_conversion(k, ind[i]);
        if(status != rocgraph_status_success)
        {
            return status;
        }
    }
    return rocgraph_status_success;
}

template <typename I, typename J>
rocgraph_status rocgraph_importer_mlcsr::import_graph_gebsx(rocgraph_direction*  dir,
                                                            rocgraph_direction*  dirb,
                                                            J*                   mb,
                                                            J*                   nb,
                                                            I*                   nnzb,
                                                            J*                   block_dim_row,
                                                            J*                   block_dim_column,
                                                            rocgraph_index_base* base)
{
    return rocgraph_status_not_implemented;
}

template <typename T, typename I, typename J>
rocgraph_status rocgraph_importer_mlcsr::import_graph_gebsx(I* ptr, J* ind, T* val)
{
    return rocgraph_status_not_implemented;
}

template <typename I>
rocgraph_status
    rocgraph_importer_mlcsr::import_graph_coo(I* m, I* n, int64_t* nnz, rocgraph_index_base* base)
{
    return rocgraph_status_not_implemented;
}

template <typename T, typename I>
rocgraph_status rocgraph_importer_mlcsr::import_graph_coo(I* row_ind, I* col_ind, T* val)
{
    return rocgraph_status_not_implemented;
}

#define INSTANTIATE_TIJ(T, I, J)                                                    \
    template rocgraph_status rocgraph_importer_mlcsr::import_graph_csx(I*, J*, T*); \
    template rocgraph_status rocgraph_importer_mlcsr::import_graph_gebsx(I*, J*, T*)

#define INSTANTIATE_TI(T, I)                                            \
    template rocgraph_status rocgraph_importer_mlcsr::import_graph_coo( \
        I* row_ind, I* col_ind, T* val)

#define INSTANTIATE_I(I)                                                \
    template rocgraph_status rocgraph_importer_mlcsr::import_graph_coo( \
        I* m, I* n, int64_t* nnz, rocgraph_index_base* base)

#define INSTANTIATE_IJ(I, J)                                              \
    template rocgraph_status rocgraph_importer_mlcsr::import_graph_csx(   \
        rocgraph_direction*, J*, J*, I*, rocgraph_index_base*);           \
    template rocgraph_status rocgraph_importer_mlcsr::import_graph_gebsx( \
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
