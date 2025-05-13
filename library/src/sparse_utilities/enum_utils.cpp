/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "enum_utils.hpp"
#include "handle.h"
namespace rocgraph
{

    template <>
    bool enum_utils::is_invalid(rocgraph_pointer_mode value)
    {
        switch(value)
        {
        case rocgraph_pointer_mode_device:
        case rocgraph_pointer_mode_host:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    bool enum_utils::is_invalid(rocgraph_action value)
    {
        switch(value)
        {
        case rocgraph_action_symbolic:
        case rocgraph_action_numeric:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    bool enum_utils::is_invalid(rocgraph_spmat_attribute value)
    {
        switch(value)
        {
        case rocgraph_spmat_fill_mode:
        case rocgraph_spmat_diag_type:
        case rocgraph_spmat_matrix_type:
        case rocgraph_spmat_storage_mode:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    bool enum_utils::is_invalid(rocgraph_diag_type value)
    {
        switch(value)
        {
        case rocgraph_diag_type_unit:
        case rocgraph_diag_type_non_unit:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    bool enum_utils::is_invalid(rocgraph_fill_mode value_)
    {
        switch(value_)
        {
        case rocgraph_fill_mode_lower:
        case rocgraph_fill_mode_upper:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    bool enum_utils::is_invalid(rocgraph_storage_mode value_)
    {
        switch(value_)
        {
        case rocgraph_storage_mode_sorted:
        case rocgraph_storage_mode_unsorted:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    bool enum_utils::is_invalid(rocgraph_index_base value_)
    {
        switch(value_)
        {
        case rocgraph_index_base_zero:
        case rocgraph_index_base_one:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    bool enum_utils::is_invalid(rocgraph_matrix_type value_)
    {
        switch(value_)
        {
        case rocgraph_matrix_type_general:
        case rocgraph_matrix_type_symmetric:
        case rocgraph_matrix_type_hermitian:
        case rocgraph_matrix_type_triangular:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    bool enum_utils::is_invalid(rocgraph_direction value_)
    {
        switch(value_)
        {
        case rocgraph_direction_row:
        case rocgraph_direction_column:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    bool enum_utils::is_invalid(rocgraph_operation value_)
    {
        switch(value_)
        {
        case rocgraph_operation_none:
        case rocgraph_operation_transpose:
        case rocgraph_operation_conjugate_transpose:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    bool enum_utils::is_invalid(rocgraph_indextype value_)
    {
        switch(value_)
        {
        case rocgraph_indextype_u16:
        case rocgraph_indextype_i32:
        case rocgraph_indextype_i64:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    bool enum_utils::is_invalid(rocgraph_datatype value_)
    {
        switch(value_)
        {
        case rocgraph_datatype_f32_r:
        case rocgraph_datatype_f64_r:
        case rocgraph_datatype_i8_r:
        case rocgraph_datatype_u8_r:
        case rocgraph_datatype_i32_r:
        case rocgraph_datatype_u32_r:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    bool enum_utils::is_invalid(rocgraph_order value_)
    {
        switch(value_)
        {
        case rocgraph_order_row:
        case rocgraph_order_column:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    bool enum_utils::is_invalid(rocgraph_spmv_alg value_)
    {
        switch(value_)
        {
        case rocgraph_spmv_alg_default:
        case rocgraph_spmv_alg_coo:
        case rocgraph_spmv_alg_csr_adaptive:
        case rocgraph_spmv_alg_csr_stream:
        case rocgraph_spmv_alg_coo_atomic:
        case rocgraph_spmv_alg_csr_lrb:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    bool enum_utils::is_invalid(rocgraph_spmv_stage value_)
    {
        switch(value_)
        {
        case rocgraph_spmv_stage_buffer_size:
        case rocgraph_spmv_stage_preprocess:
        case rocgraph_spmv_stage_compute:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    bool enum_utils::is_invalid(rocgraph_spmm_alg value_)
    {
        switch(value_)
        {
        case rocgraph_spmm_alg_default:
        case rocgraph_spmm_alg_csr:
        case rocgraph_spmm_alg_coo_segmented:
        case rocgraph_spmm_alg_coo_atomic:
        case rocgraph_spmm_alg_csr_row_split:
        case rocgraph_spmm_alg_csr_nnz_split:
        case rocgraph_spmm_alg_csr_merge_path:
        case rocgraph_spmm_alg_coo_segmented_atomic:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    bool enum_utils::is_invalid(rocgraph_spmm_stage value_)
    {
        switch(value_)
        {
        case rocgraph_spmm_stage_buffer_size:
        case rocgraph_spmm_stage_preprocess:
        case rocgraph_spmm_stage_compute:
        {
            return false;
        }
        }
        return true;
    };
}
