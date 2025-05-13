/*! \file */

// Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph.hpp"

#include "rocgraph_enum.hpp"

constexpr rocgraph_matrix_type_t::value_t
    rocgraph_matrix_type_t::values[rocgraph_matrix_type_t::nvalues];

constexpr rocgraph_operation_t::value_t rocgraph_operation_t::values[rocgraph_operation_t::nvalues];

constexpr rocgraph_storage_mode_t::value_t
    rocgraph_storage_mode_t::values[rocgraph_storage_mode_t::nvalues];

std::ostream& operator<<(std::ostream& out, const rocgraph_operation& v)
{
    out << rocgraph_operation2string(v);
    return out;
}

std::ostream& operator<<(std::ostream& out, const rocgraph_direction& v)
{
    out << rocgraph_direction2string(v);
    return out;
}
