/*! \file */

// Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_importer_matrixmarket.hpp"
#include <stdio.h>
rocgraph_importer_matrixmarket::rocgraph_importer_matrixmarket(const std::string& filename_)
    : m_filename(filename_)
{
}

/* ============================================================================================ */
/*! \brief  Read matrix from mtx file in COO format */
static inline void read_mtx_value(std::istringstream& is, int64_t& row, int64_t& col, int8_t& val)
{
    is >> row >> col >> val;
}

static inline void read_mtx_value(std::istringstream& is, int64_t& row, int64_t& col, float& val)
{
    is >> row >> col >> val;
}

static inline void read_mtx_value(std::istringstream& is, int64_t& row, int64_t& col, double& val)
{
    is >> row >> col >> val;
}

template <typename I, typename J>
rocgraph_status

    rocgraph_importer_matrixmarket::import_graph_csx(
        rocgraph_direction* dir, J* m, J* n, I* nnz, rocgraph_index_base* base)
{
    return rocgraph_status_not_implemented;
}

template <typename T, typename I, typename J>
rocgraph_status rocgraph_importer_matrixmarket::import_graph_csx(I* ptr, J* ind, T* val)
{
    return rocgraph_status_not_implemented;
}

template <typename I, typename J>
rocgraph_status rocgraph_importer_matrixmarket::import_graph_gebsx(rocgraph_direction* dir,
                                                                   rocgraph_direction* dirb,
                                                                   J*                  mb,
                                                                   J*                  nb,
                                                                   I*                  nnzb,
                                                                   J* block_dim_row,
                                                                   J* block_dim_column,
                                                                   rocgraph_index_base* base)
{
    return rocgraph_status_not_implemented;
}

template <typename T, typename I, typename J>
rocgraph_status rocgraph_importer_matrixmarket::import_graph_gebsx(I* ptr, J* ind, T* val)
{
    return rocgraph_status_not_implemented;
}

template <typename I>
rocgraph_status rocgraph_importer_matrixmarket::import_graph_coo(I*                   m,
                                                                 I*                   n,
                                                                 int64_t*             nnz,
                                                                 rocgraph_index_base* base)
{
    char line[1024];
    f = fopen(this->m_filename.c_str(), "r");
    if(!f)
    {
        missing_file_error_message(this->m_filename.c_str());
        return rocgraph_status_internal_error;
    }
    // Check for banner
    if(!fgets(line, 1024, f))
    {
        throw rocgraph_status_internal_error;
    }

    char banner[16];
    char array[16];
    char coord[16];
    char type[16];

    // Extract banner
    if(sscanf(line, "%15s %15s %15s %15s %15s", banner, array, coord, this->m_data, type) != 5)
    {
        throw rocgraph_status_internal_error;
    }

    // Convert to lower case
    for(char* p = array; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char* p = coord; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char* p = this->m_data; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char* p = type; *p != '\0'; *p = tolower(*p), p++)
        ;

    // Check banner
    if(strncmp(line, "%%MatrixMarket", 14) != 0)
    {
        throw rocgraph_status_internal_error;
    }

    // Check array type
    if(strcmp(array, "matrix") != 0)
    {
        throw rocgraph_status_internal_error;
    }

    // Check coord
    if(strcmp(coord, "coordinate") != 0)
    {
        throw rocgraph_status_internal_error;
    }

    // Check this->m_data
    if(strcmp(this->m_data, "real") != 0 && strcmp(this->m_data, "integer") != 0
       && strcmp(this->m_data, "pattern") != 0 && strcmp(this->m_data, "complex") != 0)
    {
        throw rocgraph_status_internal_error;
    }

    // Check type
    if(strcmp(type, "general") != 0 && strcmp(type, "symmetric") != 0)
    {
        throw rocgraph_status_internal_error;
    }

    // Symmetric flag
    this->m_symm = !strcmp(type, "symmetric");

    // Skip comments
    while(fgets(line, 1024, f))
    {
        if(line[0] != '%')
        {
            break;
        }
    }

    // Read dimensions
    I snnz;

    int inrow;
    int incol;
    int innz;

    sscanf(line, "%d %d %d", &inrow, &incol, &innz);

    rocgraph_status status;
    status = rocgraph_type_conversion(inrow, m[0]);
    if(status != rocgraph_status_success)
        return status;

    status = rocgraph_type_conversion(incol, n[0]);
    if(status != rocgraph_status_success)
        return status;

    status = rocgraph_type_conversion(innz, snnz);
    if(status != rocgraph_status_success)
        return status;

    if(this->m_symm)
    {
        //
        //
        // We need to count how many diagonal elements are in the file.
        //
        //

        //
        // Record position.
        //
        fpos_t pos;
        if(0 != fgetpos(this->f, &pos))
        {
            throw rocgraph_status_internal_error;
        }

        //
        // Count diagonal coefficients.
        //
        I num_diagonal_coefficients = 0;
        while(fgets(line, 1024, f))
        {
            int32_t irow{};
            int32_t icol{};
            sscanf(line, "%d %d", &irow, &icol);
            if(irow == icol)
            {
                ++num_diagonal_coefficients;
            }
        }

        //
        // Set position.
        //
        if(0 != fsetpos(this->f, &pos))
        {
            throw rocgraph_status_internal_error;
        }

        //
        // Now calculate the right number of coefficients.
        //
        snnz = (snnz - num_diagonal_coefficients) * 2 + num_diagonal_coefficients;
    }

    status = rocgraph_type_conversion(snnz, nnz[0]);
    if(status != rocgraph_status_success)
        return status;

    base[0]     = rocgraph_index_base_one;
    this->m_nnz = snnz;
    return rocgraph_status_success;
}

template <typename T, typename I>
rocgraph_status rocgraph_importer_matrixmarket::import_graph_coo(I* row_ind, I* col_ind, T* val)
{
    char           line[1024];
    const size_t   nnz = this->m_nnz;
    std::vector<I> unsorted_row(nnz);
    std::vector<I> unsorted_col(nnz);
    std::vector<T> unsorted_val(nnz);

    // Read entries
    I idx = 0;
    while(fgets(line, 1024, f))
    {
        if(idx >= nnz)
        {
            throw rocgraph_status_internal_error;
        }

        int64_t irow{};
        int64_t icol{};
        T       ival;

        std::istringstream ss(line);

        if(!strcmp(this->m_data, "pattern"))
        {
            ss >> irow >> icol;
            ival = static_cast<T>(1);
        }
        else
        {
            read_mtx_value(ss, irow, icol, ival);
        }

        unsorted_row[idx] = (I)irow;
        unsorted_col[idx] = (I)icol;
        unsorted_val[idx] = ival;

        ++idx;
        if(this->m_symm && irow != icol)
        {
            if(idx >= nnz)
            {
                throw rocgraph_status_internal_error;
            }

            unsorted_row[idx] = (I)icol;
            unsorted_col[idx] = (I)irow;
            unsorted_val[idx] = ival;
            ++idx;
        }
    }
    fclose(f);

    // Sort by row and column index
    std::vector<I> perm(nnz);
    for(I i = 0; i < nnz; ++i)
    {
        perm[i] = i;
    }

    std::sort(perm.begin(), perm.end(), [&](const I& a, const I& b) {
        if(unsorted_row[a] < unsorted_row[b])
        {
            return true;
        }
        else if(unsorted_row[a] == unsorted_row[b])
        {
            return (unsorted_col[a] < unsorted_col[b]);
        }
        else
        {
            return false;
        }
    });

    for(I i = 0; i < nnz; ++i)
    {
        row_ind[i] = unsorted_row[perm[i]];
    }
    for(I i = 0; i < nnz; ++i)
    {
        col_ind[i] = unsorted_col[perm[i]];
    }
    for(I i = 0; i < nnz; ++i)
    {
        val[i] = unsorted_val[perm[i]];
    }

    return rocgraph_status_success;
}

#define INSTANTIATE_TIJ(T, I, J)                                                           \
    template rocgraph_status rocgraph_importer_matrixmarket::import_graph_csx(I*, J*, T*); \
    template rocgraph_status rocgraph_importer_matrixmarket::import_graph_gebsx(I*, J*, T*)

#define INSTANTIATE_TI(T, I)                                                   \
    template rocgraph_status rocgraph_importer_matrixmarket::import_graph_coo( \
        I* row_ind, I* col_ind, T* val)

#define INSTANTIATE_I(I)                                                       \
    template rocgraph_status rocgraph_importer_matrixmarket::import_graph_coo( \
        I* m, I* n, int64_t* nnz, rocgraph_index_base* base)

#define INSTANTIATE_IJ(I, J)                                                     \
    template rocgraph_status rocgraph_importer_matrixmarket::import_graph_csx(   \
        rocgraph_direction*, J*, J*, I*, rocgraph_index_base*);                  \
    template rocgraph_status rocgraph_importer_matrixmarket::import_graph_gebsx( \
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
