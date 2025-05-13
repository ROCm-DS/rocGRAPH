// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "to_string.hpp"
#include "control.h"

const char* rocgraph::to_string(rocgraph_status status)
{
    switch(status)
    {
    case rocgraph_status_unknown_error:
        return "unknown error";
    case rocgraph_status_invalid_input:
        return "invalid input";
    case rocgraph_status_unsupported_type_combination:
        return "unsupported type combination";
    case rocgraph_status_success:
        return "success";
    case rocgraph_status_invalid_handle:
        return "invalid handle";
    case rocgraph_status_not_implemented:
        return "not implemented";
    case rocgraph_status_invalid_pointer:
        return "invalid pointer";
    case rocgraph_status_invalid_size:
        return "invalid size";
    case rocgraph_status_memory_error:
        return "memory error";
    case rocgraph_status_internal_error:
        return "internal error";
    case rocgraph_status_invalid_value:
        return "invalid value";
    case rocgraph_status_arch_mismatch:
        return "arch mismatch";
    case rocgraph_status_not_initialized:
        return "not initialized";
    case rocgraph_status_type_mismatch:
        return "type mismatch";
    case rocgraph_status_requires_sorted_storage:
        return "requires sorted storage";
    case rocgraph_status_thrown_exception:
        return "thrown exception";
    case rocgraph_status_continue:
        return "continue";
    }
    THROW_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_value);
}

#define CASE(C) \
    case C:     \
        return #C

const char* rocgraph::to_string(rocgraph_pointer_mode value)
{
    switch(value)
    {
        CASE(rocgraph_pointer_mode_device);
        CASE(rocgraph_pointer_mode_host);
    }
    THROW_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_value);
};

const char* rocgraph::to_string(rocgraph_spmat_attribute value)
{
    switch(value)
    {
        CASE(rocgraph_spmat_fill_mode);
        CASE(rocgraph_spmat_diag_type);
        CASE(rocgraph_spmat_matrix_type);
        CASE(rocgraph_spmat_storage_mode);
    }
    THROW_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_value);
};

const char* rocgraph::to_string(rocgraph_diag_type value)
{
    switch(value)
    {
        CASE(rocgraph_diag_type_unit);
        CASE(rocgraph_diag_type_non_unit);
    }
    THROW_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_value);
};

const char* rocgraph::to_string(rocgraph_fill_mode value_)
{
    switch(value_)
    {
        CASE(rocgraph_fill_mode_lower);
        CASE(rocgraph_fill_mode_upper);
    }
    THROW_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_value);
};

const char* rocgraph::to_string(rocgraph_storage_mode value_)
{
    switch(value_)
    {
        CASE(rocgraph_storage_mode_sorted);
        CASE(rocgraph_storage_mode_unsorted);
    }
    THROW_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_value);
};

const char* rocgraph::to_string(rocgraph_index_base value_)
{
    switch(value_)
    {
        CASE(rocgraph_index_base_zero);
        CASE(rocgraph_index_base_one);
    }
    THROW_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_value);
};

const char* rocgraph::to_string(rocgraph_matrix_type value_)
{
    switch(value_)
    {
        CASE(rocgraph_matrix_type_general);
        CASE(rocgraph_matrix_type_symmetric);
        CASE(rocgraph_matrix_type_hermitian);
        CASE(rocgraph_matrix_type_triangular);
    }
    THROW_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_value);
};

const char* rocgraph::to_string(rocgraph_direction value_)
{
    switch(value_)
    {
        CASE(rocgraph_direction_row);
        CASE(rocgraph_direction_column);
    }
    THROW_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_value);
};

const char* rocgraph::to_string(rocgraph_action value_)
{
    switch(value_)
    {
        CASE(rocgraph_action_symbolic);
        CASE(rocgraph_action_numeric);
    }
    THROW_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_value);
};

const char* rocgraph::to_string(rocgraph_operation value_)
{
    switch(value_)
    {
        CASE(rocgraph_operation_none);
        CASE(rocgraph_operation_transpose);
        CASE(rocgraph_operation_conjugate_transpose);
    }
    THROW_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_value);
};

const char* rocgraph::to_string(rocgraph_indextype value_)
{
    switch(value_)
    {
        CASE(rocgraph_indextype_u16);
        CASE(rocgraph_indextype_i32);
        CASE(rocgraph_indextype_i64);
    }
    THROW_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_value);
};

const char* rocgraph::to_string(rocgraph_datatype value_)
{
    switch(value_)
    {
        CASE(rocgraph_datatype_f32_r);
        CASE(rocgraph_datatype_f64_r);
        CASE(rocgraph_datatype_i8_r);
        CASE(rocgraph_datatype_u8_r);
        CASE(rocgraph_datatype_i32_r);
        CASE(rocgraph_datatype_u32_r);
    }
    THROW_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_value);
};

const char* rocgraph::to_string(rocgraph_order value_)
{
    switch(value_)
    {
        CASE(rocgraph_order_row);
        CASE(rocgraph_order_column);
    }
    THROW_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_value);
};

const char* rocgraph::to_string(rocgraph_spmv_alg value_)
{
    switch(value_)
    {
        CASE(rocgraph_spmv_alg_default);
        CASE(rocgraph_spmv_alg_coo);
        CASE(rocgraph_spmv_alg_csr_adaptive);
        CASE(rocgraph_spmv_alg_csr_stream);
        CASE(rocgraph_spmv_alg_coo_atomic);
        CASE(rocgraph_spmv_alg_csr_lrb);
    }
    THROW_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_value);
};

const char* rocgraph::to_string(rocgraph_spmv_stage value_)
{
    switch(value_)
    {
        CASE(rocgraph_spmv_stage_buffer_size);
        CASE(rocgraph_spmv_stage_preprocess);
        CASE(rocgraph_spmv_stage_compute);
    }
    THROW_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_value);
};

const char* rocgraph::to_string(rocgraph_spmm_alg value_)
{
    switch(value_)
    {
        CASE(rocgraph_spmm_alg_default);
        CASE(rocgraph_spmm_alg_csr);
        CASE(rocgraph_spmm_alg_coo_segmented);
        CASE(rocgraph_spmm_alg_coo_atomic);
        CASE(rocgraph_spmm_alg_csr_row_split);
        CASE(rocgraph_spmm_alg_csr_nnz_split);
        CASE(rocgraph_spmm_alg_csr_merge_path);
        CASE(rocgraph_spmm_alg_coo_segmented_atomic);
    }
    THROW_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_value);
};

const char* rocgraph::to_string(rocgraph_spmm_stage value_)
{
    switch(value_)
    {
        CASE(rocgraph_spmm_stage_buffer_size);
        CASE(rocgraph_spmm_stage_preprocess);
        CASE(rocgraph_spmm_stage_compute);
    }
    THROW_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_value);
};

const char* rocgraph::to_string(rocgraph_format value_)
{
    switch(value_)
    {
        CASE(rocgraph_format_coo);
        CASE(rocgraph_format_coo_aos);
        CASE(rocgraph_format_csr);
        CASE(rocgraph_format_csc);
    }
    THROW_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_value);
};

#undef CASE
