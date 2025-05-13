/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "internal/types/rocgraph_status.h"
#include "sparse_utility_types.h"

namespace rocgraph
{
    const char* to_string(rocgraph_action value_);
    const char* to_string(rocgraph_datatype value_);
    const char* to_string(rocgraph_diag_type value_);
    const char* to_string(rocgraph_direction value_);
    const char* to_string(rocgraph_fill_mode value_);
    const char* to_string(rocgraph_format value_);
    const char* to_string(rocgraph_indextype value_);
    const char* to_string(rocgraph_index_base value_);
    const char* to_string(rocgraph_matrix_type type);
    const char* to_string(rocgraph_matrix_type value_);
    const char* to_string(rocgraph_operation value_);
    const char* to_string(rocgraph_order value_);
    const char* to_string(rocgraph_pointer_mode value);
    const char* to_string(rocgraph_spmat_attribute value);
    const char* to_string(rocgraph_spmv_alg value_);
    const char* to_string(rocgraph_spmv_stage value_);
    const char* to_string(rocgraph_spmm_alg value_);
    const char* to_string(rocgraph_spmm_stage value_);
    const char* to_string(rocgraph_status status);
    const char* to_string(rocgraph_storage_mode value_);
}
