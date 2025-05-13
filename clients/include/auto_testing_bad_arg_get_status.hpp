/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/*! \file
 *  \brief auto_testing_bad_arg_get_status.hpp maps invalid status with types.
 */

#pragma once
#include "rocgraph-types.h"
#include "rocgraph_sparse_utility.hpp"

template <typename T>
inline rocgraph_status auto_testing_bad_arg_get_status(T& p)
{
    return rocgraph_status_invalid_pointer;
}

template <typename T>
inline rocgraph_status auto_testing_bad_arg_get_status(const T& p)
{
    return rocgraph_status_invalid_pointer;
}

template <>
inline rocgraph_status auto_testing_bad_arg_get_status(rocgraph_handle& p)
{
    return rocgraph_status_invalid_handle;
}

template <>
inline rocgraph_status auto_testing_bad_arg_get_status(rocgraph_spmat_descr& p)
{
    return rocgraph_status_invalid_pointer;
}

template <>
inline rocgraph_status auto_testing_bad_arg_get_status(rocgraph_spvec_descr& p)
{
    return rocgraph_status_invalid_pointer;
}

template <>
inline rocgraph_status auto_testing_bad_arg_get_status(rocgraph_dnmat_descr& p)
{
    return rocgraph_status_invalid_pointer;
}

template <>
inline rocgraph_status auto_testing_bad_arg_get_status(rocgraph_dnvec_descr& p)
{
    return rocgraph_status_invalid_pointer;
}

template <>
inline rocgraph_status auto_testing_bad_arg_get_status(int32_t& p)
{
    return rocgraph_status_invalid_size;
}

template <>
inline rocgraph_status auto_testing_bad_arg_get_status(size_t& p)
{
    return rocgraph_status_invalid_size;
}

template <>
inline rocgraph_status auto_testing_bad_arg_get_status(int64_t& p)
{
    return rocgraph_status_invalid_size;
}

template <>
inline rocgraph_status auto_testing_bad_arg_get_status(rocgraph_operation& p)
{
    return rocgraph_status_invalid_value;
}

template <>
inline rocgraph_status auto_testing_bad_arg_get_status(rocgraph_order& p)
{
    return rocgraph_status_invalid_value;
}

template <>
inline rocgraph_status auto_testing_bad_arg_get_status(rocgraph_index_base& p)
{
    return rocgraph_status_invalid_value;
}

template <>
inline rocgraph_status auto_testing_bad_arg_get_status(rocgraph_matrix_type& p)
{
    return rocgraph_status_invalid_value;
}

template <>
inline rocgraph_status auto_testing_bad_arg_get_status(rocgraph_fill_mode& p)
{
    return rocgraph_status_invalid_value;
}

template <>
inline rocgraph_status auto_testing_bad_arg_get_status(rocgraph_storage_mode& p)
{
    return rocgraph_status_invalid_value;
}

template <>
inline rocgraph_status auto_testing_bad_arg_get_status(rocgraph_indextype& p)
{
    return rocgraph_status_invalid_value;
}

template <>
inline rocgraph_status auto_testing_bad_arg_get_status(rocgraph_datatype& p)
{
    return rocgraph_status_invalid_value;
}

template <>
inline rocgraph_status auto_testing_bad_arg_get_status(rocgraph_direction& p)
{
    return rocgraph_status_invalid_value;
}

template <>
inline rocgraph_status auto_testing_bad_arg_get_status(rocgraph_action& p)
{
    return rocgraph_status_invalid_value;
}
