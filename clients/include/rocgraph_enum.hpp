/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/*! \file
 *  \brief rocgraph_enum.hpp provides common testing utilities.
 */

#pragma once
#ifndef ROCGRAPH_ENUM_HPP
#define ROCGRAPH_ENUM_HPP

#include "rocgraph_test.hpp"
#include <hip/hip_runtime_api.h>
#include <vector>

struct rocgraph_matrix_type_t
{
    using value_t                     = rocgraph_matrix_type;
    static constexpr uint32_t nvalues = 4;
    // clang-format off
  static constexpr value_t  values[nvalues] = {rocgraph_matrix_type_general,
                                               rocgraph_matrix_type_symmetric,
                                               rocgraph_matrix_type_hermitian,
                                               rocgraph_matrix_type_triangular};
    // clang-format on
};

struct rocgraph_operation_t
{
    using value_t                     = rocgraph_operation;
    static constexpr uint32_t nvalues = 3;
    // clang-format off
    static constexpr value_t  values[nvalues] = {rocgraph_operation_none,
                                                 rocgraph_operation_transpose,
                                                 rocgraph_operation_conjugate_transpose};
    // clang-format on
};

struct rocgraph_storage_mode_t
{
    using value_t                     = rocgraph_storage_mode;
    static constexpr uint32_t nvalues = 2;
    static constexpr value_t  values[nvalues]
        = {rocgraph_storage_mode_sorted, rocgraph_storage_mode_unsorted};
};

std::ostream& operator<<(std::ostream& out, const rocgraph_operation& v);
std::ostream& operator<<(std::ostream& out, const rocgraph_direction& v);

struct rocgraph_datatype_t
{
    using value_t = rocgraph_datatype;
    template <typename T>
    static inline rocgraph_datatype get();
};

template <>
inline rocgraph_datatype rocgraph_datatype_t::get<float>()
{
    return rocgraph_datatype_f32_r;
}
template <>
inline rocgraph_datatype rocgraph_datatype_t::get<double>()
{
    return rocgraph_datatype_f64_r;
}

#endif // ROCGRAPH_ENUM_HPP
