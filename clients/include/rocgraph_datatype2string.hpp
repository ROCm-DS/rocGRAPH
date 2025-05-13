/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef ROCGRAPH_DATATYPE2STRING_HPP
#define ROCGRAPH_DATATYPE2STRING_HPP

#include "rocgraph_sparse_utility.hpp"
#include <rocgraph.h>
#include <string>

#include <algorithm>

typedef enum rocgraph_matrix_init_kind_
{
    rocgraph_matrix_init_kind_default  = 0,
    rocgraph_matrix_init_kind_tunedavg = 1
} rocgraph_matrix_init_kind;

constexpr auto rocgraph_matrix_init_kind2string(rocgraph_matrix_init_kind matrix)
{
    switch(matrix)
    {
    case rocgraph_matrix_init_kind_default:
        return "default";
    case rocgraph_matrix_init_kind_tunedavg:
        return "tunedavg";
    }
    return "invalid";
}

typedef enum rocgraph_matrix_init_
{
    rocgraph_matrix_random          = 0, /**< Random initialization */
    rocgraph_matrix_laplace_2d      = 1, /**< Initialize 2D laplacian matrix */
    rocgraph_matrix_laplace_3d      = 2, /**< Initialize 3D laplacian matrix */
    rocgraph_matrix_file_mtx        = 3, /**< Read from matrix market file */
    rocgraph_matrix_file_smtx       = 4, /**< Read from machine learning csr file */
    rocgraph_matrix_file_rocalution = 5, /**< Read from rocalution file */
    rocgraph_matrix_zero            = 6, /**< Generates zero matrix */
    rocgraph_matrix_tridiagonal     = 7, /**< Initialize tridiagonal matrix */
    rocgraph_matrix_pentadiagonal   = 8 /**< Initialize pentadiagonal matrix */
} rocgraph_matrix_init;

constexpr auto rocgraph_matrix2string(rocgraph_matrix_init matrix)
{
    switch(matrix)
    {
    case rocgraph_matrix_random:
        return "rand";
    case rocgraph_matrix_laplace_2d:
        return "L2D";
    case rocgraph_matrix_laplace_3d:
        return "L3D";
    case rocgraph_matrix_file_mtx:
        return "mtx";
    case rocgraph_matrix_file_smtx:
        return "smtx";
    case rocgraph_matrix_file_rocalution:
        return "csr";
    case rocgraph_matrix_zero:
        return "zero";
    case rocgraph_matrix_tridiagonal:
        return "tri";
    case rocgraph_matrix_pentadiagonal:
        return "penta";
    }
    return "invalid";
}

constexpr auto rocgraph_indextype2string(rocgraph_indextype type)
{
    switch(type)
    {
    case rocgraph_indextype_u16:
        return "u16";
    case rocgraph_indextype_i32:
        return "i32";
    case rocgraph_indextype_i64:
        return "i64";
    }
    return "invalid";
}

constexpr auto rocgraph_datatype2string(rocgraph_datatype type)
{
    switch(type)
    {
    case rocgraph_datatype_f32_r:
        return "f32_r";
    case rocgraph_datatype_f64_r:
        return "f64_r";
    case rocgraph_datatype_i8_r:
        return "i8_r";
    case rocgraph_datatype_u8_r:
        return "u8_r";
    case rocgraph_datatype_i32_r:
        return "i32_r";
    case rocgraph_datatype_u32_r:
        return "u32_r";
    }
    return "invalid";
}

constexpr auto rocgraph_indexbase2string(rocgraph_index_base base)
{
    switch(base)
    {
    case rocgraph_index_base_zero:
        return "0b";
    case rocgraph_index_base_one:
        return "1b";
    }
    return "invalid";
}

constexpr auto rocgraph_operation2string(rocgraph_operation trans)
{
    switch(trans)
    {
    case rocgraph_operation_none:
        return "NT";
    case rocgraph_operation_transpose:
        return "T";
    case rocgraph_operation_conjugate_transpose:
        return "CT";
    }
    return "invalid";
}

constexpr auto rocgraph_matrixtype2string(rocgraph_matrix_type type)
{
    switch(type)
    {
    case rocgraph_matrix_type_general:
        return "general";
    case rocgraph_matrix_type_symmetric:
        return "symmetric";
    case rocgraph_matrix_type_hermitian:
        return "hermitian";
    case rocgraph_matrix_type_triangular:
        return "triangular";
    }
    return "invalid";
}

constexpr auto rocgraph_diagtype2string(rocgraph_diag_type diag)
{
    switch(diag)
    {
    case rocgraph_diag_type_non_unit:
        return "ND";
    case rocgraph_diag_type_unit:
        return "UD";
    }
    return "invalid";
}

constexpr auto rocgraph_fillmode2string(rocgraph_fill_mode uplo)
{
    switch(uplo)
    {
    case rocgraph_fill_mode_lower:
        return "L";
    case rocgraph_fill_mode_upper:
        return "U";
    }
    return "invalid";
}

constexpr auto rocgraph_storagemode2string(rocgraph_storage_mode storage)
{
    switch(storage)
    {
    case rocgraph_storage_mode_sorted:
        return "sorted";
    case rocgraph_storage_mode_unsorted:
        return "unsorted";
    }
    return "invalid";
}

constexpr auto rocgraph_action2string(rocgraph_action action)
{
    switch(action)
    {
    case rocgraph_action_symbolic:
        return "sym";
    case rocgraph_action_numeric:
        return "num";
    }
    return "invalid";
}

constexpr auto rocgraph_partition2string(rocgraph_hyb_partition part)
{
    switch(part)
    {
    case rocgraph_hyb_partition_auto:
        return "auto";
    case rocgraph_hyb_partition_user:
        return "user";
    case rocgraph_hyb_partition_max:
        return "max";
    }
    return "invalid";
}

constexpr auto rocgraph_analysis2string(rocgraph_analysis_policy policy)
{
    switch(policy)
    {
    case rocgraph_analysis_policy_reuse:
        return "reuse";
    case rocgraph_analysis_policy_force:
        return "force";
    }
    return "invalid";
}

constexpr auto rocgraph_solve2string(rocgraph_solve_policy policy)
{
    switch(policy)
    {
    case rocgraph_solve_policy_auto:
        return "auto";
    }
    return "invalid";
}

constexpr auto rocgraph_direction2string(rocgraph_direction direction)
{
    switch(direction)
    {
    case rocgraph_direction_row:
        return "row";
    case rocgraph_direction_column:
        return "column";
    }
    return "invalid";
}

constexpr auto rocgraph_order2string(rocgraph_order order)
{
    switch(order)
    {
    case rocgraph_order_row:
        return "row";
    case rocgraph_order_column:
        return "col";
    }
    return "invalid";
}

constexpr auto rocgraph_format2string(rocgraph_format format)
{
    switch(format)
    {
    case rocgraph_format_coo:
        return "coo";
    case rocgraph_format_coo_aos:
        return "coo_aos";
    case rocgraph_format_csr:
        return "csr";
    case rocgraph_format_bsr:
        return "bsr";
    case rocgraph_format_csc:
        return "csc";
    case rocgraph_format_ell:
        return "ell";
    case rocgraph_format_bell:
        return "bell";
    }
    return "invalid";
}

// Return a string without '/' or '\\'
inline std::string rocgraph_filename2string(const std::string& filename)
{
    std::string result(filename);
    std::replace(result.begin(), result.end(), '/', '_');
    std::replace(result.begin(), result.end(), '\\', '_');
    return result;
}

#endif // ROCGRAPH_DATATYPE2STRING_HPP
