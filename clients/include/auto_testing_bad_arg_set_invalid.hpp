/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/*! \file
 *  \brief auto_testing_bad_arg_set_invalid.hpp maps bad values with types.
 */

#pragma once
#include "rocgraph-types.h"
//
// PROVIDE TEMPLATE FOR AUTO TESTING BAD ARGUMENTS
//

template <typename T>
inline void auto_testing_bad_arg_set_invalid(T& p);

template <typename T>
inline void auto_testing_bad_arg_set_invalid(T*& p)
{
    p = nullptr;
}

template <>
inline void auto_testing_bad_arg_set_invalid(int32_t& p)
{
    p = -1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(size_t& p)
{
}

template <>
inline void auto_testing_bad_arg_set_invalid(int64_t& p)
{
    p = -1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(float& p)
{
    p = static_cast<float>(-1);
}

template <>
inline void auto_testing_bad_arg_set_invalid(double& p)
{
    p = static_cast<double>(-1);
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocgraph_operation& p)
{
    p = (rocgraph_operation)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocgraph_order& p)
{
    p = (rocgraph_order)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocgraph_index_base& p)
{
    p = (rocgraph_index_base)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocgraph_matrix_type& p)
{
    p = (rocgraph_matrix_type)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocgraph_fill_mode& p)
{
    p = (rocgraph_fill_mode)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocgraph_storage_mode& p)
{
    p = (rocgraph_storage_mode)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocgraph_indextype& p)
{
    p = (rocgraph_indextype)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocgraph_datatype& p)
{
    p = (rocgraph_datatype)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocgraph_analysis_policy& p)
{
    p = (rocgraph_analysis_policy)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocgraph_direction& p)
{
    p = (rocgraph_direction)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocgraph_action& p)
{
    p = (rocgraph_action)-1;
}
