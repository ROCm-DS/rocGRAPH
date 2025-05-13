// Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "common.h"
#include "control.h"
#include "handle.h"
#include "rocgraph.h"
#include "utility.h"
#include <iomanip>
#include <map>

#include <hip/hip_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************************
 * \brief Get rocGRAPH status enum name as a string
 *******************************************************************************/
const char* rocgraph_get_status_name(rocgraph_status status)
{
    switch(status)
    {
    case rocgraph_status_success:
        return "rocgraph_status_success";
    case rocgraph_status_unknown_error:
        return "rocgraph_status_unknown_error";
    case rocgraph_status_invalid_input:
        return "rocgraph_status_invalid_input";
    case rocgraph_status_unsupported_type_combination:
        return "rocgraph_status_unsupported_type_combination";
    case rocgraph_status_invalid_handle:
        return "rocgraph_status_invalid_handle";
    case rocgraph_status_not_implemented:
        return "rocgraph_status_not_implemented";
    case rocgraph_status_invalid_pointer:
        return "rocgraph_status_invalid_pointer";
    case rocgraph_status_invalid_size:
        return "rocgraph_status_invalid_size";
    case rocgraph_status_memory_error:
        return "rocgraph_status_memory_error";
    case rocgraph_status_internal_error:
        return "rocgraph_status_internal_error";
    case rocgraph_status_invalid_value:
        return "rocgraph_status_invalid_value";
    case rocgraph_status_arch_mismatch:
        return "rocgraph_status_arch_mismatch";
    case rocgraph_status_not_initialized:
        return "rocgraph_status_not_initialized";
    case rocgraph_status_type_mismatch:
        return "rocgraph_status_type_mismatch";
    case rocgraph_status_requires_sorted_storage:
        return "rocgraph_status_requires_sorted_storage";
    case rocgraph_status_thrown_exception:
        return "rocgraph_status_thrown_exception";
    case rocgraph_status_continue:
        return "rocgraph_status_continue";
    }

    return "Unrecognized status code";
}

/********************************************************************************
 * \brief Get rocGRAPH status enum description as a string
 *******************************************************************************/
const char* rocgraph_get_status_description(rocgraph_status status)
{
    switch(status)
    {
    case rocgraph_status_unknown_error:
        return "the error is not categorized";
    case rocgraph_status_invalid_input:
        return "the input is invalid";
    case rocgraph_status_unsupported_type_combination:
        return "the submitted type combination is not supported";
    case rocgraph_status_success:
        return "rocgraph operation was successful";
    case rocgraph_status_invalid_handle:
        return "handle not initialized, invalid or null";
    case rocgraph_status_not_implemented:
        return "function is not implemented";
    case rocgraph_status_invalid_pointer:
        return "invalid pointer parameter";
    case rocgraph_status_invalid_size:
        return "invalid size parameter";
    case rocgraph_status_memory_error:
        return "failed memory allocation, copy, dealloc";
    case rocgraph_status_internal_error:
        return "other internal library failure";
    case rocgraph_status_invalid_value:
        return "invalid value parameter";
    case rocgraph_status_arch_mismatch:
        return "device arch is not supported";
    case rocgraph_status_not_initialized:
        return "descriptor has not been initialized";
    case rocgraph_status_type_mismatch:
        return "index types do not match";
    case rocgraph_status_requires_sorted_storage:
        return "sorted storage required";
    case rocgraph_status_thrown_exception:
        return "exception being thrown";
    case rocgraph_status_continue:
        return "nothing preventing function to proceed";
    }

    return "Unrecognized status code";
}

/********************************************************************************
 * \brief rocgraph_create_mat_descr_t is a structure holding the rocgraph matrix
 * descriptor. It must be initialized using rocgraph_create_mat_descr()
 * and the returned handle must be passed to all subsequent library function
 * calls that involve the matrix.
 * It should be destroyed at the end using rocgraph_destroy_mat_descr().
 *******************************************************************************/
rocgraph_status rocgraph_create_mat_descr(rocgraph_mat_descr* descr)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    *descr = new _rocgraph_mat_descr;
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief copy matrix descriptor
 *******************************************************************************/
rocgraph_status rocgraph_copy_mat_descr(rocgraph_mat_descr dest, const rocgraph_mat_descr src)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, dest);
    ROCGRAPH_CHECKARG_POINTER(1, src);
    ROCGRAPH_CHECKARG(1, src, (src == dest), rocgraph_status_invalid_pointer);

    dest->fill_mode    = src->fill_mode;
    dest->diag_type    = src->diag_type;
    dest->type         = src->type;
    dest->base         = src->base;
    dest->storage_mode = src->storage_mode;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief destroy matrix descriptor
 *******************************************************************************/
rocgraph_status rocgraph_destroy_mat_descr(rocgraph_mat_descr descr)
try
{
    delete descr;
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief Set the index base of the matrix descriptor.
 *******************************************************************************/
rocgraph_status rocgraph_set_mat_index_base(rocgraph_mat_descr descr, rocgraph_index_base base)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG_ENUM(1, base);
    descr->base = base;
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief Returns the index base of the matrix descriptor.
 *******************************************************************************/
rocgraph_index_base rocgraph_get_mat_index_base(const rocgraph_mat_descr descr)
{
    // If descriptor is invalid, default index base is returned
    if(descr == nullptr)
    {
        return rocgraph_index_base_zero;
    }
    return descr->base;
}

/********************************************************************************
 * \brief Set the matrix type of the matrix descriptor.
 *******************************************************************************/
rocgraph_status rocgraph_set_mat_type(rocgraph_mat_descr descr, rocgraph_matrix_type type)
try
{

    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG_ENUM(1, type);

    descr->type = type;
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief Returns the matrix type of the matrix descriptor.
 *******************************************************************************/
rocgraph_matrix_type rocgraph_get_mat_type(const rocgraph_mat_descr descr)
{
    // If descriptor is invalid, default matrix type is returned
    if(descr == nullptr)
    {
        return rocgraph_matrix_type_general;
    }
    return descr->type;
}

rocgraph_status rocgraph_set_mat_fill_mode(rocgraph_mat_descr descr, rocgraph_fill_mode fill_mode)
try
{

    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG_ENUM(1, fill_mode);

    descr->fill_mode = fill_mode;
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

rocgraph_fill_mode rocgraph_get_mat_fill_mode(const rocgraph_mat_descr descr)
{
    // If descriptor is invalid, default fill mode is returned
    if(descr == nullptr)
    {
        return rocgraph_fill_mode_lower;
    }
    return descr->fill_mode;
}

rocgraph_status rocgraph_set_mat_diag_type(rocgraph_mat_descr descr, rocgraph_diag_type diag_type)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG_ENUM(1, diag_type);
    descr->diag_type = diag_type;
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

rocgraph_diag_type rocgraph_get_mat_diag_type(const rocgraph_mat_descr descr)
{
    // If descriptor is invalid, default diagonal type is returned
    if(descr == nullptr)
    {
        return rocgraph_diag_type_non_unit;
    }
    return descr->diag_type;
}

rocgraph_status rocgraph_set_mat_storage_mode(rocgraph_mat_descr    descr,
                                              rocgraph_storage_mode storage_mode)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG_ENUM(1, storage_mode);
    descr->storage_mode = storage_mode;
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

rocgraph_storage_mode rocgraph_get_mat_storage_mode(const rocgraph_mat_descr descr)
{
    // If descriptor is invalid, default fill mode is returned
    if(descr == nullptr)
    {
        return rocgraph_storage_mode_sorted;
    }
    return descr->storage_mode;
}

/********************************************************************************
 * \brief rocgraph_mat_info is a structure holding the matrix info data that is
 * gathered during the analysis routines. It must be initialized by calling
 * rocgraph_create_mat_info() and the returned info structure must be passed
 * to all subsequent function calls that require additional information. It
 * should be destroyed at the end using rocgraph_destroy_mat_info().
 *******************************************************************************/
rocgraph_status rocgraph_create_mat_info(rocgraph_mat_info* info)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, info);
    *info = new _rocgraph_mat_info;
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief Copy mat info.
 *******************************************************************************/
rocgraph_status rocgraph_copy_mat_info(rocgraph_mat_info dest, const rocgraph_mat_info src)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, dest);
    ROCGRAPH_CHECKARG_POINTER(1, src);
    ROCGRAPH_CHECKARG(1, src, (src == dest), rocgraph_status_invalid_pointer);
    if(src->csrmv_info != nullptr)
    {
        if(dest->csrmv_info == nullptr)
        {
            RETURN_IF_ROCGRAPH_ERROR(rocgraph::create_csrmv_info(&dest->csrmv_info));
        }
        RETURN_IF_ROCGRAPH_ERROR(rocgraph::copy_csrmv_info(dest->csrmv_info, src->csrmv_info));
    }
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief Destroy mat info.
 *******************************************************************************/
rocgraph_status rocgraph_destroy_mat_info(rocgraph_mat_info info)
try
{
    if(info == nullptr)
    {
        return rocgraph_status_success;
    }

    // Clear csrmv info struct
    if(info->csrmv_info != nullptr)
    {
        RETURN_IF_ROCGRAPH_ERROR(rocgraph::destroy_csrmv_info(info->csrmv_info));
    }
    try
    {
        delete info;
    }
    catch(const rocgraph_status& status)
    {
        return status;
    }
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_color_info is a structure holding the color info data that is
 * gathered during the analysis routines. It must be initialized by calling
 * rocgraph_create_color_info() and the returned info structure must be passed
 * to all subsequent function calls that require additional information. It
 * should be destroyed at the end using rocgraph_destroy_color_info().
 *******************************************************************************/
rocgraph_status rocgraph_create_color_info(rocgraph_color_info* info)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, info);
    *info = new _rocgraph_color_info;
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief Copy color info.
 *******************************************************************************/
rocgraph_status rocgraph_copy_color_info(rocgraph_color_info dest, const rocgraph_color_info src)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, dest);
    ROCGRAPH_CHECKARG_POINTER(1, src);
    ROCGRAPH_CHECKARG(1, src, (src == dest), rocgraph_status_invalid_pointer);

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief Destroy color info.
 *******************************************************************************/
rocgraph_status rocgraph_destroy_color_info(rocgraph_color_info info)
try
{
    if(info == nullptr)
    {
        return rocgraph_status_success;
    }
    delete info;
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_create_coo_descr creates a descriptor holding the COO matrix
 * data, sizes and properties. It must be called prior to all subsequent library
 * function calls that involve graph matrices. It should be destroyed at the end
 * using rocgraph_destroy_spmat_descr(). All data pointers remain valid.
 *******************************************************************************/
rocgraph_status rocgraph_create_coo_descr(rocgraph_spmat_descr* descr,
                                          int64_t               rows,
                                          int64_t               cols,
                                          int64_t               nnz,
                                          void*                 coo_row_ind,
                                          void*                 coo_col_ind,
                                          void*                 coo_val,
                                          rocgraph_indextype    idx_type,
                                          rocgraph_index_base   idx_base,
                                          rocgraph_datatype     data_type)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG_SIZE(1, rows);
    ROCGRAPH_CHECKARG_SIZE(2, cols);
    ROCGRAPH_CHECKARG_SIZE(3, nnz);
    ROCGRAPH_CHECKARG(3, nnz, (nnz > rows * cols), rocgraph_status_invalid_size);
    ROCGRAPH_CHECKARG_ARRAY(4, nnz, coo_row_ind);
    ROCGRAPH_CHECKARG_ARRAY(5, nnz, coo_col_ind);
    ROCGRAPH_CHECKARG_ARRAY(6, nnz, coo_val);
    ROCGRAPH_CHECKARG_ENUM(7, idx_type);
    ROCGRAPH_CHECKARG_ENUM(8, idx_base);
    ROCGRAPH_CHECKARG_ENUM(9, data_type);

    *descr = new _rocgraph_spmat_descr;

    (*descr)->init = true;

    (*descr)->rows = rows;
    (*descr)->cols = cols;
    (*descr)->nnz  = nnz;

    (*descr)->row_data = coo_row_ind;
    (*descr)->col_data = coo_col_ind;
    (*descr)->val_data = coo_val;

    (*descr)->const_row_data = coo_row_ind;
    (*descr)->const_col_data = coo_col_ind;
    (*descr)->const_val_data = coo_val;

    (*descr)->row_type  = idx_type;
    (*descr)->col_type  = idx_type;
    (*descr)->data_type = data_type;

    (*descr)->idx_base = idx_base;
    (*descr)->format   = rocgraph_format_coo;

    RETURN_IF_ROCGRAPH_ERROR(rocgraph_create_mat_descr(&(*descr)->descr));
    RETURN_IF_ROCGRAPH_ERROR(rocgraph_create_mat_info(&(*descr)->info));
    RETURN_IF_ROCGRAPH_ERROR(rocgraph_set_mat_index_base((*descr)->descr, idx_base));

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

rocgraph_status rocgraph_create_const_coo_descr(rocgraph_const_spmat_descr* descr,
                                                int64_t                     rows,
                                                int64_t                     cols,
                                                int64_t                     nnz,
                                                const void*                 coo_row_ind,
                                                const void*                 coo_col_ind,
                                                const void*                 coo_val,
                                                rocgraph_indextype          idx_type,
                                                rocgraph_index_base         idx_base,
                                                rocgraph_datatype           data_type)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG_SIZE(1, rows);
    ROCGRAPH_CHECKARG_SIZE(2, cols);
    ROCGRAPH_CHECKARG_SIZE(3, nnz);
    ROCGRAPH_CHECKARG(3, nnz, (nnz > rows * cols), rocgraph_status_invalid_size);
    ROCGRAPH_CHECKARG_ARRAY(4, nnz, coo_row_ind);
    ROCGRAPH_CHECKARG_ARRAY(5, nnz, coo_col_ind);
    ROCGRAPH_CHECKARG_ARRAY(6, nnz, coo_val);
    ROCGRAPH_CHECKARG_ENUM(7, idx_type);
    ROCGRAPH_CHECKARG_ENUM(8, idx_base);
    ROCGRAPH_CHECKARG_ENUM(9, data_type);

    rocgraph_spmat_descr new_descr = new _rocgraph_spmat_descr;

    new_descr->init = true;

    new_descr->rows = rows;
    new_descr->cols = cols;
    new_descr->nnz  = nnz;

    new_descr->row_data = nullptr;
    new_descr->col_data = nullptr;
    new_descr->val_data = nullptr;

    new_descr->const_row_data = coo_row_ind;
    new_descr->const_col_data = coo_col_ind;
    new_descr->const_val_data = coo_val;

    new_descr->row_type  = idx_type;
    new_descr->col_type  = idx_type;
    new_descr->data_type = data_type;

    new_descr->idx_base = idx_base;
    new_descr->format   = rocgraph_format_coo;

    RETURN_IF_ROCGRAPH_ERROR(rocgraph_create_mat_descr(&new_descr->descr));
    RETURN_IF_ROCGRAPH_ERROR(rocgraph_create_mat_info(&new_descr->info));

    // Initialize descriptor
    RETURN_IF_ROCGRAPH_ERROR(rocgraph_set_mat_index_base(new_descr->descr, idx_base));

    *descr = new_descr;
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_create_coo_aos_descr creates a descriptor holding the COO matrix
 * data, sizes and properties where the row pointer and column indices are stored
 * using array of structure (AoS) format. It must be called prior to all subsequent
 * library function calls that involve graph matrices. It should be destroyed at
 * the end using rocgraph_destroy_spmat_descr(). All data pointers remain valid.
 *******************************************************************************/
rocgraph_status rocgraph_create_coo_aos_descr(rocgraph_spmat_descr* descr,
                                              int64_t               rows,
                                              int64_t               cols,
                                              int64_t               nnz,
                                              void*                 coo_ind,
                                              void*                 coo_val,
                                              rocgraph_indextype    idx_type,
                                              rocgraph_index_base   idx_base,
                                              rocgraph_datatype     data_type)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG_SIZE(1, rows);
    ROCGRAPH_CHECKARG_SIZE(2, cols);
    ROCGRAPH_CHECKARG_SIZE(3, nnz);
    ROCGRAPH_CHECKARG(3, nnz, (nnz > rows * cols), rocgraph_status_invalid_size);
    ROCGRAPH_CHECKARG_ARRAY(4, nnz, coo_ind);
    ROCGRAPH_CHECKARG_ARRAY(5, nnz, coo_val);
    ROCGRAPH_CHECKARG_ENUM(6, idx_type);
    ROCGRAPH_CHECKARG_ENUM(7, idx_base);
    ROCGRAPH_CHECKARG_ENUM(8, data_type);

    *descr = new _rocgraph_spmat_descr;

    (*descr)->init = true;

    (*descr)->rows = rows;
    (*descr)->cols = cols;
    (*descr)->nnz  = nnz;

    (*descr)->ind_data = coo_ind;
    (*descr)->val_data = coo_val;

    (*descr)->const_ind_data = coo_ind;
    (*descr)->const_val_data = coo_val;

    (*descr)->row_type  = idx_type;
    (*descr)->col_type  = idx_type;
    (*descr)->data_type = data_type;

    (*descr)->idx_base = idx_base;
    (*descr)->format   = rocgraph_format_coo_aos;

    RETURN_IF_ROCGRAPH_ERROR(rocgraph_create_mat_descr(&(*descr)->descr));
    RETURN_IF_ROCGRAPH_ERROR(rocgraph_create_mat_info(&(*descr)->info));

    // Initialize descriptor
    RETURN_IF_ROCGRAPH_ERROR(rocgraph_set_mat_index_base((*descr)->descr, idx_base));

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_create_csr_descr creates a descriptor holding the CSR matrix
 * data, sizes and properties. It must be called prior to all subsequent library
 * function calls that involve graph matrices. It should be destroyed at the end
 * using rocgraph_destroy_spmat_descr(). All data pointers remain valid.
 *******************************************************************************/
rocgraph_status rocgraph_create_csr_descr(rocgraph_spmat_descr* descr,
                                          int64_t               rows,
                                          int64_t               cols,
                                          int64_t               nnz,
                                          void*                 csr_row_ptr,
                                          void*                 csr_col_ind,
                                          void*                 csr_val,
                                          rocgraph_indextype    row_ptr_type,
                                          rocgraph_indextype    col_ind_type,
                                          rocgraph_index_base   idx_base,
                                          rocgraph_datatype     data_type)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG_SIZE(1, rows);
    ROCGRAPH_CHECKARG_SIZE(2, cols);
    ROCGRAPH_CHECKARG_SIZE(3, nnz);
    ROCGRAPH_CHECKARG(3, nnz, (nnz > rows * cols), rocgraph_status_invalid_size);

    //
    // SWDEV-340500, this is a non-sense.
    // cugraph parity behavior should be fixed in hipgraph, not here.
    //
    //    ROCGRAPH_CHECKARG(4, (rows > 0 && nnz > 0 && csr_row_ptr == nullptr), csr_row_ptr, rocgraph_status_invalid_pointer);
    ROCGRAPH_CHECKARG_ARRAY(4, rows, csr_row_ptr);
    ROCGRAPH_CHECKARG_ARRAY(5, nnz, csr_col_ind);
    ROCGRAPH_CHECKARG_ARRAY(6, nnz, csr_val);
    ROCGRAPH_CHECKARG_ENUM(7, row_ptr_type);
    ROCGRAPH_CHECKARG_ENUM(8, col_ind_type);
    ROCGRAPH_CHECKARG_ENUM(9, idx_base);
    ROCGRAPH_CHECKARG_ENUM(10, data_type);

    *descr = new _rocgraph_spmat_descr;

    (*descr)->init = true;

    (*descr)->rows = rows;
    (*descr)->cols = cols;
    (*descr)->nnz  = nnz;

    (*descr)->row_data = csr_row_ptr;
    (*descr)->col_data = csr_col_ind;
    (*descr)->val_data = csr_val;

    (*descr)->const_row_data = csr_row_ptr;
    (*descr)->const_col_data = csr_col_ind;
    (*descr)->const_val_data = csr_val;

    (*descr)->row_type  = row_ptr_type;
    (*descr)->col_type  = col_ind_type;
    (*descr)->data_type = data_type;

    (*descr)->idx_base = idx_base;
    (*descr)->format   = rocgraph_format_csr;

    RETURN_IF_ROCGRAPH_ERROR(rocgraph_create_mat_descr(&(*descr)->descr));
    RETURN_IF_ROCGRAPH_ERROR(rocgraph_create_mat_info(&(*descr)->info));

    // Initialize descriptor
    RETURN_IF_ROCGRAPH_ERROR(rocgraph_set_mat_index_base((*descr)->descr, idx_base));

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

rocgraph_status rocgraph_create_const_csr_descr(rocgraph_const_spmat_descr* descr,
                                                int64_t                     rows,
                                                int64_t                     cols,
                                                int64_t                     nnz,
                                                const void*                 csr_row_ptr,
                                                const void*                 csr_col_ind,
                                                const void*                 csr_val,
                                                rocgraph_indextype          row_ptr_type,
                                                rocgraph_indextype          col_ind_type,
                                                rocgraph_index_base         idx_base,
                                                rocgraph_datatype           data_type)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG_SIZE(1, rows);
    ROCGRAPH_CHECKARG_SIZE(2, cols);
    ROCGRAPH_CHECKARG_SIZE(3, nnz);
    ROCGRAPH_CHECKARG(3, nnz, (nnz > rows * cols), rocgraph_status_invalid_size);

    //
    // SWDEV-340500, this is a non-sense.
    // cugraph parity behavior should be fixed in hipgraph, not here.
    //
    //    ROCGRAPH_CHECKARG(4, (rows > 0 && nnz > 0 && csr_row_ptr == nullptr), csr_row_ptr, rocgraph_status_invalid_pointer);
    ROCGRAPH_CHECKARG_ARRAY(4, rows, csr_row_ptr);

    ROCGRAPH_CHECKARG_ARRAY(5, nnz, csr_col_ind);
    ROCGRAPH_CHECKARG_ARRAY(6, nnz, csr_val);
    ROCGRAPH_CHECKARG_ENUM(7, row_ptr_type);
    ROCGRAPH_CHECKARG_ENUM(8, col_ind_type);
    ROCGRAPH_CHECKARG_ENUM(9, idx_base);
    ROCGRAPH_CHECKARG_ENUM(10, data_type);

    rocgraph_spmat_descr new_descr = new _rocgraph_spmat_descr;

    new_descr->init = true;

    new_descr->rows = rows;
    new_descr->cols = cols;
    new_descr->nnz  = nnz;

    new_descr->row_data = nullptr;
    new_descr->col_data = nullptr;
    new_descr->val_data = nullptr;

    new_descr->const_row_data = csr_row_ptr;
    new_descr->const_col_data = csr_col_ind;
    new_descr->const_val_data = csr_val;

    new_descr->row_type  = row_ptr_type;
    new_descr->col_type  = col_ind_type;
    new_descr->data_type = data_type;

    new_descr->idx_base = idx_base;
    new_descr->format   = rocgraph_format_csr;

    RETURN_IF_ROCGRAPH_ERROR(rocgraph_create_mat_descr(&new_descr->descr));
    RETURN_IF_ROCGRAPH_ERROR(rocgraph_create_mat_info(&new_descr->info));

    // Initialize descriptor
    RETURN_IF_ROCGRAPH_ERROR(rocgraph_set_mat_index_base(new_descr->descr, idx_base));

    *descr = new_descr;
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_create_csc_descr creates a descriptor holding the CSC matrix
 * data, sizes and properties. It must be called prior to all subsequent library
 * function calls that involve graph matrices. It should be destroyed at the end
 * using rocgraph_destroy_spmat_descr(). All data pointers remain valid.
 *******************************************************************************/
rocgraph_status rocgraph_create_csc_descr(rocgraph_spmat_descr* descr,
                                          int64_t               rows,
                                          int64_t               cols,
                                          int64_t               nnz,
                                          void*                 csc_col_ptr,
                                          void*                 csc_row_ind,
                                          void*                 csc_val,
                                          rocgraph_indextype    col_ptr_type,
                                          rocgraph_indextype    row_ind_type,
                                          rocgraph_index_base   idx_base,
                                          rocgraph_datatype     data_type)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG_SIZE(1, rows);
    ROCGRAPH_CHECKARG_SIZE(2, cols);
    ROCGRAPH_CHECKARG_SIZE(3, nnz);
    ROCGRAPH_CHECKARG(3, nnz, (nnz > rows * cols), rocgraph_status_invalid_size);
    ROCGRAPH_CHECKARG_ARRAY(4, cols, csc_col_ptr);
    ROCGRAPH_CHECKARG_ARRAY(5, nnz, csc_row_ind);
    ROCGRAPH_CHECKARG_ARRAY(6, nnz, csc_val);
    ROCGRAPH_CHECKARG_ENUM(7, col_ptr_type);
    ROCGRAPH_CHECKARG_ENUM(8, row_ind_type);
    ROCGRAPH_CHECKARG_ENUM(9, idx_base);
    ROCGRAPH_CHECKARG_ENUM(10, data_type);
    *descr = new _rocgraph_spmat_descr;

    (*descr)->init = true;

    (*descr)->rows = rows;
    (*descr)->cols = cols;
    (*descr)->nnz  = nnz;

    (*descr)->row_data = csc_row_ind;
    (*descr)->col_data = csc_col_ptr;
    (*descr)->val_data = csc_val;

    (*descr)->const_row_data = csc_row_ind;
    (*descr)->const_col_data = csc_col_ptr;
    (*descr)->const_val_data = csc_val;

    (*descr)->row_type  = row_ind_type;
    (*descr)->col_type  = col_ptr_type;
    (*descr)->data_type = data_type;

    (*descr)->idx_base = idx_base;
    (*descr)->format   = rocgraph_format_csc;

    RETURN_IF_ROCGRAPH_ERROR(rocgraph_create_mat_descr(&(*descr)->descr));
    RETURN_IF_ROCGRAPH_ERROR(rocgraph_create_mat_info(&(*descr)->info));

    // Initialize descriptor
    RETURN_IF_ROCGRAPH_ERROR(rocgraph_set_mat_index_base((*descr)->descr, idx_base));

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

rocgraph_status rocgraph_create_const_csc_descr(rocgraph_const_spmat_descr* descr,
                                                int64_t                     rows,
                                                int64_t                     cols,
                                                int64_t                     nnz,
                                                const void*                 csc_col_ptr,
                                                const void*                 csc_row_ind,
                                                const void*                 csc_val,
                                                rocgraph_indextype          col_ptr_type,
                                                rocgraph_indextype          row_ind_type,
                                                rocgraph_index_base         idx_base,
                                                rocgraph_datatype           data_type)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG_SIZE(1, rows);
    ROCGRAPH_CHECKARG_SIZE(2, cols);
    ROCGRAPH_CHECKARG_SIZE(3, nnz);
    ROCGRAPH_CHECKARG(3, nnz, (nnz > rows * cols), rocgraph_status_invalid_size);
    ROCGRAPH_CHECKARG_ARRAY(4, cols, csc_col_ptr);
    ROCGRAPH_CHECKARG_ARRAY(5, nnz, csc_row_ind);
    ROCGRAPH_CHECKARG_ARRAY(6, nnz, csc_val);
    ROCGRAPH_CHECKARG_ENUM(7, col_ptr_type);
    ROCGRAPH_CHECKARG_ENUM(8, row_ind_type);
    ROCGRAPH_CHECKARG_ENUM(9, idx_base);
    ROCGRAPH_CHECKARG_ENUM(10, data_type);

    rocgraph_spmat_descr new_descr = new _rocgraph_spmat_descr;

    new_descr->init = true;

    new_descr->rows = rows;
    new_descr->cols = cols;
    new_descr->nnz  = nnz;

    new_descr->row_data = nullptr;
    new_descr->col_data = nullptr;
    new_descr->val_data = nullptr;

    new_descr->const_row_data = csc_row_ind;
    new_descr->const_col_data = csc_col_ptr;
    new_descr->const_val_data = csc_val;

    new_descr->row_type  = row_ind_type;
    new_descr->col_type  = col_ptr_type;
    new_descr->data_type = data_type;

    new_descr->idx_base = idx_base;
    new_descr->format   = rocgraph_format_csc;

    RETURN_IF_ROCGRAPH_ERROR(rocgraph_create_mat_descr(&new_descr->descr));
    RETURN_IF_ROCGRAPH_ERROR(rocgraph_create_mat_info(&new_descr->info));

    // Initialize descriptor
    RETURN_IF_ROCGRAPH_ERROR(rocgraph_set_mat_index_base(new_descr->descr, idx_base));

    *descr = new_descr;
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_destroy_spmat_descr destroys a graph matrix descriptor.
 *******************************************************************************/
rocgraph_status rocgraph_destroy_spmat_descr(rocgraph_const_spmat_descr descr)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        // Do nothing
        return rocgraph_status_success;
    }

    RETURN_IF_ROCGRAPH_ERROR(rocgraph_destroy_mat_descr(descr->descr));
    RETURN_IF_ROCGRAPH_ERROR(rocgraph_destroy_mat_info(descr->info));

    delete descr;
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_coo_get returns the graph COO matrix data, sizes and
 * properties.
 *******************************************************************************/
rocgraph_status rocgraph_coo_get(const rocgraph_spmat_descr descr,
                                 int64_t*                   rows,
                                 int64_t*                   cols,
                                 int64_t*                   nnz,
                                 void**                     coo_row_ind,
                                 void**                     coo_col_ind,
                                 void**                     coo_val,
                                 rocgraph_indextype*        idx_type,
                                 rocgraph_index_base*       idx_base,
                                 rocgraph_datatype*         data_type)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, rows);
    ROCGRAPH_CHECKARG_POINTER(2, cols);
    ROCGRAPH_CHECKARG_POINTER(3, nnz);
    ROCGRAPH_CHECKARG_POINTER(4, coo_row_ind);
    ROCGRAPH_CHECKARG_POINTER(5, coo_col_ind);
    ROCGRAPH_CHECKARG_POINTER(6, coo_val);
    ROCGRAPH_CHECKARG_POINTER(7, idx_type);
    ROCGRAPH_CHECKARG_POINTER(8, idx_base);
    ROCGRAPH_CHECKARG_POINTER(9, data_type);

    *rows = descr->rows;
    *cols = descr->cols;
    *nnz  = descr->nnz;

    *coo_row_ind = descr->row_data;
    *coo_col_ind = descr->col_data;
    *coo_val     = descr->val_data;

    *idx_type  = descr->row_type;
    *idx_base  = descr->idx_base;
    *data_type = descr->data_type;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

rocgraph_status rocgraph_const_coo_get(rocgraph_const_spmat_descr descr,
                                       int64_t*                   rows,
                                       int64_t*                   cols,
                                       int64_t*                   nnz,
                                       const void**               coo_row_ind,
                                       const void**               coo_col_ind,
                                       const void**               coo_val,
                                       rocgraph_indextype*        idx_type,
                                       rocgraph_index_base*       idx_base,
                                       rocgraph_datatype*         data_type)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, rows);
    ROCGRAPH_CHECKARG_POINTER(2, cols);
    ROCGRAPH_CHECKARG_POINTER(3, nnz);
    ROCGRAPH_CHECKARG_POINTER(4, coo_row_ind);
    ROCGRAPH_CHECKARG_POINTER(5, coo_col_ind);
    ROCGRAPH_CHECKARG_POINTER(6, coo_val);
    ROCGRAPH_CHECKARG_POINTER(7, idx_type);
    ROCGRAPH_CHECKARG_POINTER(8, idx_base);
    ROCGRAPH_CHECKARG_POINTER(9, data_type);

    *rows = descr->rows;
    *cols = descr->cols;
    *nnz  = descr->nnz;

    *coo_row_ind = descr->const_row_data;
    *coo_col_ind = descr->const_col_data;
    *coo_val     = descr->const_val_data;

    *idx_type  = descr->row_type;
    *idx_base  = descr->idx_base;
    *data_type = descr->data_type;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_coo_aos_get returns the graph COO (AoS) matrix data, sizes and
 * properties.
 *******************************************************************************/
rocgraph_status rocgraph_coo_aos_get(const rocgraph_spmat_descr descr,
                                     int64_t*                   rows,
                                     int64_t*                   cols,
                                     int64_t*                   nnz,
                                     void**                     coo_ind,
                                     void**                     coo_val,
                                     rocgraph_indextype*        idx_type,
                                     rocgraph_index_base*       idx_base,
                                     rocgraph_datatype*         data_type)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, rows);
    ROCGRAPH_CHECKARG_POINTER(2, cols);
    ROCGRAPH_CHECKARG_POINTER(3, nnz);
    ROCGRAPH_CHECKARG_POINTER(4, coo_ind);
    ROCGRAPH_CHECKARG_POINTER(5, coo_val);
    ROCGRAPH_CHECKARG_POINTER(6, idx_type);
    ROCGRAPH_CHECKARG_POINTER(7, idx_base);
    ROCGRAPH_CHECKARG_POINTER(8, data_type);

    *rows = descr->rows;
    *cols = descr->cols;
    *nnz  = descr->nnz;

    *coo_ind = descr->ind_data;
    *coo_val = descr->val_data;

    *idx_type  = descr->row_type;
    *idx_base  = descr->idx_base;
    *data_type = descr->data_type;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

rocgraph_status rocgraph_const_coo_aos_get(rocgraph_const_spmat_descr descr,
                                           int64_t*                   rows,
                                           int64_t*                   cols,
                                           int64_t*                   nnz,
                                           const void**               coo_ind,
                                           const void**               coo_val,
                                           rocgraph_indextype*        idx_type,
                                           rocgraph_index_base*       idx_base,
                                           rocgraph_datatype*         data_type)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, rows);
    ROCGRAPH_CHECKARG_POINTER(2, cols);
    ROCGRAPH_CHECKARG_POINTER(3, nnz);
    ROCGRAPH_CHECKARG_POINTER(4, coo_ind);
    ROCGRAPH_CHECKARG_POINTER(5, coo_val);
    ROCGRAPH_CHECKARG_POINTER(6, idx_type);
    ROCGRAPH_CHECKARG_POINTER(7, idx_base);
    ROCGRAPH_CHECKARG_POINTER(8, data_type);

    *rows = descr->rows;
    *cols = descr->cols;
    *nnz  = descr->nnz;

    *coo_ind = descr->const_ind_data;
    *coo_val = descr->const_val_data;

    *idx_type  = descr->row_type;
    *idx_base  = descr->idx_base;
    *data_type = descr->data_type;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_csr_get returns the graph CSR matrix data, sizes and
 * properties.
 *******************************************************************************/
rocgraph_status rocgraph_csr_get(const rocgraph_spmat_descr descr,
                                 int64_t*                   rows,
                                 int64_t*                   cols,
                                 int64_t*                   nnz,
                                 void**                     csr_row_ptr,
                                 void**                     csr_col_ind,
                                 void**                     csr_val,
                                 rocgraph_indextype*        row_ptr_type,
                                 rocgraph_indextype*        col_ind_type,
                                 rocgraph_index_base*       idx_base,
                                 rocgraph_datatype*         data_type)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, rows);
    ROCGRAPH_CHECKARG_POINTER(2, cols);
    ROCGRAPH_CHECKARG_POINTER(3, nnz);
    ROCGRAPH_CHECKARG_POINTER(4, csr_row_ptr);
    ROCGRAPH_CHECKARG_POINTER(5, csr_col_ind);
    ROCGRAPH_CHECKARG_POINTER(6, csr_val);
    ROCGRAPH_CHECKARG_POINTER(7, row_ptr_type);
    ROCGRAPH_CHECKARG_POINTER(8, col_ind_type);
    ROCGRAPH_CHECKARG_POINTER(9, idx_base);
    ROCGRAPH_CHECKARG_POINTER(10, data_type);

    *rows = descr->rows;
    *cols = descr->cols;
    *nnz  = descr->nnz;

    *csr_row_ptr = descr->row_data;
    *csr_col_ind = descr->col_data;
    *csr_val     = descr->val_data;

    *row_ptr_type = descr->row_type;
    *col_ind_type = descr->col_type;
    *idx_base     = descr->idx_base;
    *data_type    = descr->data_type;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

rocgraph_status rocgraph_const_csr_get(rocgraph_const_spmat_descr descr,
                                       int64_t*                   rows,
                                       int64_t*                   cols,
                                       int64_t*                   nnz,
                                       const void**               csr_row_ptr,
                                       const void**               csr_col_ind,
                                       const void**               csr_val,
                                       rocgraph_indextype*        row_ptr_type,
                                       rocgraph_indextype*        col_ind_type,
                                       rocgraph_index_base*       idx_base,
                                       rocgraph_datatype*         data_type)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, rows);
    ROCGRAPH_CHECKARG_POINTER(2, cols);
    ROCGRAPH_CHECKARG_POINTER(3, nnz);
    ROCGRAPH_CHECKARG_POINTER(4, csr_row_ptr);
    ROCGRAPH_CHECKARG_POINTER(5, csr_col_ind);
    ROCGRAPH_CHECKARG_POINTER(6, csr_val);
    ROCGRAPH_CHECKARG_POINTER(7, row_ptr_type);
    ROCGRAPH_CHECKARG_POINTER(8, col_ind_type);
    ROCGRAPH_CHECKARG_POINTER(9, idx_base);
    ROCGRAPH_CHECKARG_POINTER(10, data_type);

    *rows = descr->rows;
    *cols = descr->cols;
    *nnz  = descr->nnz;

    *csr_row_ptr = descr->const_row_data;
    *csr_col_ind = descr->const_col_data;
    *csr_val     = descr->const_val_data;

    *row_ptr_type = descr->row_type;
    *col_ind_type = descr->col_type;
    *idx_base     = descr->idx_base;
    *data_type    = descr->data_type;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_csc_get returns the graph CSC matrix data, sizes and
 * properties.
 *******************************************************************************/
rocgraph_status rocgraph_csc_get(const rocgraph_spmat_descr descr,
                                 int64_t*                   rows,
                                 int64_t*                   cols,
                                 int64_t*                   nnz,
                                 void**                     csc_col_ptr,
                                 void**                     csc_row_ind,
                                 void**                     csc_val,
                                 rocgraph_indextype*        col_ptr_type,
                                 rocgraph_indextype*        row_ind_type,
                                 rocgraph_index_base*       idx_base,
                                 rocgraph_datatype*         data_type)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, rows);
    ROCGRAPH_CHECKARG_POINTER(2, cols);
    ROCGRAPH_CHECKARG_POINTER(3, nnz);
    ROCGRAPH_CHECKARG_POINTER(4, csc_col_ptr);
    ROCGRAPH_CHECKARG_POINTER(5, csc_row_ind);
    ROCGRAPH_CHECKARG_POINTER(6, csc_val);
    ROCGRAPH_CHECKARG_POINTER(7, col_ptr_type);
    ROCGRAPH_CHECKARG_POINTER(8, row_ind_type);
    ROCGRAPH_CHECKARG_POINTER(9, idx_base);
    ROCGRAPH_CHECKARG_POINTER(10, data_type);

    *rows = descr->rows;
    *cols = descr->cols;
    *nnz  = descr->nnz;

    *csc_col_ptr = descr->col_data;
    *csc_row_ind = descr->row_data;
    *csc_val     = descr->val_data;

    *col_ptr_type = descr->col_type;
    *row_ind_type = descr->row_type;
    *idx_base     = descr->idx_base;
    *data_type    = descr->data_type;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

rocgraph_status rocgraph_const_csc_get(rocgraph_const_spmat_descr descr,
                                       int64_t*                   rows,
                                       int64_t*                   cols,
                                       int64_t*                   nnz,
                                       const void**               csc_col_ptr,
                                       const void**               csc_row_ind,
                                       const void**               csc_val,
                                       rocgraph_indextype*        col_ptr_type,
                                       rocgraph_indextype*        row_ind_type,
                                       rocgraph_index_base*       idx_base,
                                       rocgraph_datatype*         data_type)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, rows);
    ROCGRAPH_CHECKARG_POINTER(2, cols);
    ROCGRAPH_CHECKARG_POINTER(3, nnz);
    ROCGRAPH_CHECKARG_POINTER(4, csc_col_ptr);
    ROCGRAPH_CHECKARG_POINTER(5, csc_row_ind);
    ROCGRAPH_CHECKARG_POINTER(6, csc_val);
    ROCGRAPH_CHECKARG_POINTER(7, col_ptr_type);
    ROCGRAPH_CHECKARG_POINTER(8, row_ind_type);
    ROCGRAPH_CHECKARG_POINTER(9, idx_base);
    ROCGRAPH_CHECKARG_POINTER(10, data_type);

    *rows = descr->rows;
    *cols = descr->cols;
    *nnz  = descr->nnz;

    *csc_col_ptr = descr->const_col_data;
    *csc_row_ind = descr->const_row_data;
    *csc_val     = descr->const_val_data;

    *row_ind_type = descr->row_type;
    *col_ptr_type = descr->col_type;
    *idx_base     = descr->idx_base;
    *data_type    = descr->data_type;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_coo_set_pointers sets the graph COO matrix data pointers.
 *******************************************************************************/
rocgraph_status rocgraph_coo_set_pointers(rocgraph_spmat_descr descr,
                                          void*                coo_row_ind,
                                          void*                coo_col_ind,
                                          void*                coo_val)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, coo_row_ind);
    ROCGRAPH_CHECKARG_POINTER(2, coo_col_ind);
    ROCGRAPH_CHECKARG_POINTER(3, coo_val);

    descr->row_data = coo_row_ind;
    descr->col_data = coo_col_ind;
    descr->val_data = coo_val;

    descr->const_row_data = coo_row_ind;
    descr->const_col_data = coo_col_ind;
    descr->const_val_data = coo_val;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_coo_aos_set_pointers sets the graph COO (AoS) matrix data pointers.
 *******************************************************************************/
rocgraph_status
    rocgraph_coo_aos_set_pointers(rocgraph_spmat_descr descr, void* coo_ind, void* coo_val)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, coo_ind);
    ROCGRAPH_CHECKARG_POINTER(2, coo_val);

    descr->ind_data = coo_ind;
    descr->val_data = coo_val;

    descr->const_ind_data = coo_ind;
    descr->const_val_data = coo_val;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_csr_set_pointers sets the graph CSR matrix data pointers.
 *******************************************************************************/
rocgraph_status rocgraph_csr_set_pointers(rocgraph_spmat_descr descr,
                                          void*                csr_row_ptr,
                                          void*                csr_col_ind,
                                          void*                csr_val)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);

    ROCGRAPH_CHECKARG_POINTER(1, csr_row_ptr);
    ROCGRAPH_CHECKARG(
        2, csr_col_ind, descr->nnz > 0 && csr_col_ind == nullptr, rocgraph_status_invalid_pointer);
    ROCGRAPH_CHECKARG(
        3, csr_val, descr->nnz > 0 && csr_val == nullptr, rocgraph_status_invalid_pointer);

    // Sparsity structure might have changed, analysis is required before calling SpMV
    descr->analysed = false;

    descr->row_data = csr_row_ptr;
    descr->col_data = csr_col_ind;
    descr->val_data = csr_val;

    descr->const_row_data = csr_row_ptr;
    descr->const_col_data = csr_col_ind;
    descr->const_val_data = csr_val;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_csc_set_pointers sets the graph CSR matrix data pointers.
 *******************************************************************************/
rocgraph_status rocgraph_csc_set_pointers(rocgraph_spmat_descr descr,
                                          void*                csc_col_ptr,
                                          void*                csc_row_ind,
                                          void*                csc_val)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);

    ROCGRAPH_CHECKARG_POINTER(1, csc_col_ptr);
    ROCGRAPH_CHECKARG(
        2, csc_row_ind, descr->nnz > 0 && csc_row_ind == nullptr, rocgraph_status_invalid_pointer);
    ROCGRAPH_CHECKARG(
        3, csc_val, descr->nnz > 0 && csc_val == nullptr, rocgraph_status_invalid_pointer);

    // Sparsity structure might have changed, analysis is required before calling SpMV
    descr->analysed = false;

    descr->row_data = csc_row_ind;
    descr->col_data = csc_col_ptr;
    descr->val_data = csc_val;

    descr->const_row_data = csc_row_ind;
    descr->const_col_data = csc_col_ptr;
    descr->const_val_data = csc_val;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_spmat_get_size returns the graph matrix sizes.
 *******************************************************************************/
rocgraph_status rocgraph_spmat_get_size(rocgraph_const_spmat_descr descr,
                                        int64_t*                   rows,
                                        int64_t*                   cols,
                                        int64_t*                   nnz)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, rows);
    ROCGRAPH_CHECKARG_POINTER(2, cols);
    ROCGRAPH_CHECKARG_POINTER(3, nnz);

    *rows = descr->rows;
    *cols = descr->cols;
    *nnz  = descr->nnz;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_spmat_get_format returns the graph matrix format.
 *******************************************************************************/
rocgraph_status rocgraph_spmat_get_format(rocgraph_const_spmat_descr descr, rocgraph_format* format)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, format);

    *format = descr->format;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_spmat_get_index_base returns the graph matrix index base.
 *******************************************************************************/
rocgraph_status rocgraph_spmat_get_index_base(rocgraph_const_spmat_descr descr,
                                              rocgraph_index_base*       idx_base)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, idx_base);

    *idx_base = descr->idx_base;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_spmat_get_values returns the graph matrix value pointer.
 *******************************************************************************/
rocgraph_status rocgraph_spmat_get_values(rocgraph_spmat_descr descr, void** values)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, values);
    *values = descr->val_data;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

rocgraph_status rocgraph_const_spmat_get_values(rocgraph_const_spmat_descr descr,
                                                const void**               values)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, values);

    *values = descr->const_val_data;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_spmat_set_values sets the graph matrix value pointer.
 *******************************************************************************/
rocgraph_status rocgraph_spmat_set_values(rocgraph_spmat_descr descr, void* values)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, values);

    descr->val_data       = values;
    descr->const_val_data = values;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_spmat_get_attribute gets the graph matrix attribute.
 *******************************************************************************/
rocgraph_status rocgraph_spmat_get_attribute(rocgraph_const_spmat_descr descr,
                                             rocgraph_spmat_attribute   attribute,
                                             void*                      data,
                                             size_t                     data_size)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG_ENUM(1, attribute);
    ROCGRAPH_CHECKARG_POINTER(2, data);
    switch(attribute)
    {
    case rocgraph_spmat_fill_mode:
    {
        ROCGRAPH_CHECKARG(3,
                          data_size,
                          data_size != sizeof(rocgraph_spmat_fill_mode),
                          rocgraph_status_invalid_size);
        rocgraph_fill_mode* uplo = reinterpret_cast<rocgraph_fill_mode*>(data);
        *uplo                    = rocgraph_get_mat_fill_mode(descr->descr);
        return rocgraph_status_success;
    }
    case rocgraph_spmat_diag_type:
    {
        ROCGRAPH_CHECKARG(3,
                          data_size,
                          data_size != sizeof(rocgraph_spmat_diag_type),
                          rocgraph_status_invalid_size);
        rocgraph_diag_type* uplo = reinterpret_cast<rocgraph_diag_type*>(data);
        *uplo                    = rocgraph_get_mat_diag_type(descr->descr);
        return rocgraph_status_success;
    }

    case rocgraph_spmat_matrix_type:
    {
        ROCGRAPH_CHECKARG(3,
                          data_size,
                          data_size != sizeof(rocgraph_spmat_matrix_type),
                          rocgraph_status_invalid_size);
        rocgraph_matrix_type* matrix = reinterpret_cast<rocgraph_matrix_type*>(data);
        *matrix                      = rocgraph_get_mat_type(descr->descr);
        return rocgraph_status_success;
    }
    case rocgraph_spmat_storage_mode:
    {
        ROCGRAPH_CHECKARG(3,
                          data_size,
                          data_size != sizeof(rocgraph_spmat_storage_mode),
                          rocgraph_status_invalid_size);
        rocgraph_storage_mode* storage = reinterpret_cast<rocgraph_storage_mode*>(data);
        *storage                       = rocgraph_get_mat_storage_mode(descr->descr);
        return rocgraph_status_success;
    }
    }

    return rocgraph_status_invalid_value;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_spmat_set_attribute sets the graph matrix attribute.
 *******************************************************************************/
rocgraph_status rocgraph_spmat_set_attribute(rocgraph_spmat_descr     descr,
                                             rocgraph_spmat_attribute attribute,
                                             const void*              data,
                                             size_t                   data_size)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG_ENUM(1, attribute);
    ROCGRAPH_CHECKARG_POINTER(2, data);

    switch(attribute)
    {
    case rocgraph_spmat_fill_mode:
    {
        ROCGRAPH_CHECKARG(3,
                          data_size,
                          data_size != sizeof(rocgraph_spmat_fill_mode),
                          rocgraph_status_invalid_size);
        rocgraph_fill_mode uplo = *reinterpret_cast<const rocgraph_fill_mode*>(data);
        return rocgraph_set_mat_fill_mode(descr->descr, uplo);
    }
    case rocgraph_spmat_diag_type:
    {
        ROCGRAPH_CHECKARG(3,
                          data_size,
                          data_size != sizeof(rocgraph_spmat_diag_type),
                          rocgraph_status_invalid_size);
        rocgraph_diag_type diag = *reinterpret_cast<const rocgraph_diag_type*>(data);
        return rocgraph_set_mat_diag_type(descr->descr, diag);
    }

    case rocgraph_spmat_matrix_type:
    {
        ROCGRAPH_CHECKARG(3,
                          data_size,
                          data_size != sizeof(rocgraph_spmat_matrix_type),
                          rocgraph_status_invalid_size);
        rocgraph_matrix_type matrix = *reinterpret_cast<const rocgraph_matrix_type*>(data);
        return rocgraph_set_mat_type(descr->descr, matrix);
    }
    case rocgraph_spmat_storage_mode:
    {
        ROCGRAPH_CHECKARG(3,
                          data_size,
                          data_size != sizeof(rocgraph_spmat_storage_mode),
                          rocgraph_status_invalid_size);
        rocgraph_storage_mode storage = *reinterpret_cast<const rocgraph_storage_mode*>(data);
        return rocgraph_set_mat_storage_mode(descr->descr, storage);
    }
    }
    return rocgraph_status_invalid_value;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_create_dnvec_descr creates a descriptor holding the dense
 * vector data, size and properties. It must be called prior to all subsequent
 * library function calls that involve the dense vector. It should be destroyed
 * at the end using rocgraph_destroy_dnvec_descr(). The data pointer remains
 * valid.
 *******************************************************************************/
rocgraph_status rocgraph_create_dnvec_descr(rocgraph_dnvec_descr* descr,
                                            int64_t               size,
                                            void*                 values,
                                            rocgraph_datatype     data_type)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG_SIZE(1, size);
    ROCGRAPH_CHECKARG_ARRAY(2, size, values);
    ROCGRAPH_CHECKARG_ENUM(3, data_type);

    *descr = new _rocgraph_dnvec_descr;

    (*descr)->init = true;

    (*descr)->size         = size;
    (*descr)->values       = values;
    (*descr)->const_values = values;
    (*descr)->data_type    = data_type;
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

rocgraph_status rocgraph_create_const_dnvec_descr(rocgraph_const_dnvec_descr* descr,
                                                  int64_t                     size,
                                                  const void*                 values,
                                                  rocgraph_datatype           data_type)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG_SIZE(1, size);
    ROCGRAPH_CHECKARG_ARRAY(2, size, values);
    ROCGRAPH_CHECKARG_ENUM(3, data_type);

    rocgraph_dnvec_descr new_descr = new _rocgraph_dnvec_descr;

    new_descr->init = true;

    new_descr->size         = size;
    new_descr->values       = nullptr;
    new_descr->const_values = values;
    new_descr->data_type    = data_type;

    *descr = new_descr;
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_destroy_dnvec_descr destroys a dense vector descriptor.
 *******************************************************************************/
rocgraph_status rocgraph_destroy_dnvec_descr(rocgraph_const_dnvec_descr descr)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);

    delete descr;
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_dnvec_get returns the dense vector data, size and properties.
 *******************************************************************************/
rocgraph_status rocgraph_dnvec_get(const rocgraph_dnvec_descr descr,
                                   int64_t*                   size,
                                   void**                     values,
                                   rocgraph_datatype*         data_type)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, size);
    ROCGRAPH_CHECKARG_POINTER(2, values);
    ROCGRAPH_CHECKARG_POINTER(3, data_type);

    *size      = descr->size;
    *values    = descr->values;
    *data_type = descr->data_type;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

rocgraph_status rocgraph_const_dnvec_get(rocgraph_const_dnvec_descr descr,
                                         int64_t*                   size,
                                         const void**               values,
                                         rocgraph_datatype*         data_type)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, size);
    ROCGRAPH_CHECKARG_POINTER(2, values);
    ROCGRAPH_CHECKARG_POINTER(3, data_type);

    *size      = descr->size;
    *values    = descr->const_values;
    *data_type = descr->data_type;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_dnvec_get_values returns the dense vector value pointer.
 *******************************************************************************/
rocgraph_status rocgraph_dnvec_get_values(const rocgraph_dnvec_descr descr, void** values)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, values);

    *values = descr->values;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

rocgraph_status rocgraph_const_dnvec_get_values(rocgraph_const_dnvec_descr descr,
                                                const void**               values)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, values);

    *values = descr->const_values;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_dnvec_set_values sets the dense vector value pointer.
 *******************************************************************************/
rocgraph_status rocgraph_dnvec_set_values(rocgraph_dnvec_descr descr, void* values)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, values);
    descr->values       = values;
    descr->const_values = values;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_create_dnmat_descr creates a descriptor holding the dense
 * matrix data, size and properties. It must be called prior to all subsequent
 * library function calls that involve the dense matrix. It should be destroyed
 * at the end using rocgraph_destroy_dnmat_descr(). The data pointer remains
 * valid.
 *******************************************************************************/
rocgraph_status rocgraph_create_dnmat_descr(rocgraph_dnmat_descr* descr,
                                            int64_t               rows,
                                            int64_t               cols,
                                            int64_t               ld,
                                            void*                 values,
                                            rocgraph_datatype     data_type,
                                            rocgraph_order        order)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG_SIZE(1, rows);
    ROCGRAPH_CHECKARG_SIZE(2, cols);
    ROCGRAPH_CHECKARG_ENUM(5, data_type);
    ROCGRAPH_CHECKARG_ENUM(6, order);

    switch(order)
    {
    case rocgraph_order_row:
    {
        ROCGRAPH_CHECKARG(
            3, ld, (ld < rocgraph::max(int64_t(1), cols)), rocgraph_status_invalid_size);
        break;
    }
    case rocgraph_order_column:
    {
        ROCGRAPH_CHECKARG(
            3, ld, (ld < rocgraph::max(int64_t(1), rows)), rocgraph_status_invalid_size);
        break;
    }
    }

    ROCGRAPH_CHECKARG_ARRAY(4, int64_t(rows) * cols, values);

    *descr = new _rocgraph_dnmat_descr;

    (*descr)->init = true;

    (*descr)->rows         = rows;
    (*descr)->cols         = cols;
    (*descr)->ld           = ld;
    (*descr)->values       = values;
    (*descr)->const_values = values;
    (*descr)->data_type    = data_type;
    (*descr)->order        = order;
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

rocgraph_status rocgraph_create_const_dnmat_descr(rocgraph_const_dnmat_descr* descr,
                                                  int64_t                     rows,
                                                  int64_t                     cols,
                                                  int64_t                     ld,
                                                  const void*                 values,
                                                  rocgraph_datatype           data_type,
                                                  rocgraph_order              order)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG_SIZE(1, rows);
    ROCGRAPH_CHECKARG_SIZE(2, cols);

    switch(order)
    {
    case rocgraph_order_row:
    {
        ROCGRAPH_CHECKARG(
            3, ld, (ld < rocgraph::max(int64_t(1), cols)), rocgraph_status_invalid_size);
        break;
    }
    case rocgraph_order_column:
    {
        ROCGRAPH_CHECKARG(
            3, ld, (ld < rocgraph::max(int64_t(1), rows)), rocgraph_status_invalid_size);
        break;
    }
    }

    ROCGRAPH_CHECKARG_ARRAY(4, int64_t(rows) * cols, values);
    ROCGRAPH_CHECKARG_ENUM(5, data_type);
    ROCGRAPH_CHECKARG_ENUM(6, order);

    rocgraph_dnmat_descr new_descr = new _rocgraph_dnmat_descr;
    new_descr->init                = true;

    new_descr->rows         = rows;
    new_descr->cols         = cols;
    new_descr->ld           = ld;
    new_descr->values       = nullptr;
    new_descr->const_values = values;
    new_descr->data_type    = data_type;
    new_descr->order        = order;

    *descr = new_descr;
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_destroy_dnmat_descr destroys a dense matrix descriptor.
 *******************************************************************************/
rocgraph_status rocgraph_destroy_dnmat_descr(rocgraph_const_dnmat_descr descr)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    delete descr;
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_dnmat_get returns the dense matrix data, size and properties.
 *******************************************************************************/
rocgraph_status rocgraph_dnmat_get(const rocgraph_dnmat_descr descr,
                                   int64_t*                   rows,
                                   int64_t*                   cols,
                                   int64_t*                   ld,
                                   void**                     values,
                                   rocgraph_datatype*         data_type,
                                   rocgraph_order*            order)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, rows);
    ROCGRAPH_CHECKARG_POINTER(2, cols);
    ROCGRAPH_CHECKARG_POINTER(3, ld);
    ROCGRAPH_CHECKARG_POINTER(4, values);
    ROCGRAPH_CHECKARG_POINTER(5, data_type);
    ROCGRAPH_CHECKARG_POINTER(6, order);

    *rows      = descr->rows;
    *cols      = descr->cols;
    *ld        = descr->ld;
    *values    = descr->values;
    *data_type = descr->data_type;
    *order     = descr->order;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

rocgraph_status rocgraph_const_dnmat_get(rocgraph_const_dnmat_descr descr,
                                         int64_t*                   rows,
                                         int64_t*                   cols,
                                         int64_t*                   ld,
                                         const void**               values,
                                         rocgraph_datatype*         data_type,
                                         rocgraph_order*            order)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, rows);
    ROCGRAPH_CHECKARG_POINTER(2, cols);
    ROCGRAPH_CHECKARG_POINTER(3, ld);
    ROCGRAPH_CHECKARG_POINTER(4, values);
    ROCGRAPH_CHECKARG_POINTER(5, data_type);
    ROCGRAPH_CHECKARG_POINTER(6, order);

    *rows      = descr->rows;
    *cols      = descr->cols;
    *ld        = descr->ld;
    *values    = descr->const_values;
    *data_type = descr->data_type;
    *order     = descr->order;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_dnmat_get_values returns the dense matrix value pointer.
 *******************************************************************************/
rocgraph_status rocgraph_dnmat_get_values(const rocgraph_dnmat_descr descr, void** values)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, values);
    *values = descr->values;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

rocgraph_status rocgraph_const_dnmat_get_values(rocgraph_const_dnmat_descr descr,
                                                const void**               values)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, values);

    *values = descr->const_values;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

/********************************************************************************
 * \brief rocgraph_dnmat_set_values sets the dense matrix value pointer.
 *******************************************************************************/
rocgraph_status rocgraph_dnmat_set_values(rocgraph_dnmat_descr descr, void* values)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG(0, descr, (descr->init == false), rocgraph_status_not_initialized);
    ROCGRAPH_CHECKARG_POINTER(1, values);

    descr->values       = values;
    descr->const_values = values;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

ROCGRAPH_EXPORT
rocgraph_status rocgraph_create_csr_descr_SWDEV_453599(rocgraph_spmat_descr* descr,
                                                       int64_t               rows,
                                                       int64_t               cols,
                                                       int64_t               nnz,
                                                       void*                 csr_row_ptr,
                                                       void*                 csr_col_ind,
                                                       void*                 csr_val,
                                                       rocgraph_indextype    row_ptr_type,
                                                       rocgraph_indextype    col_ind_type,
                                                       rocgraph_index_base   idx_base,
                                                       rocgraph_datatype     data_type)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, descr);
    ROCGRAPH_CHECKARG_SIZE(1, rows);
    ROCGRAPH_CHECKARG_SIZE(2, cols);
    ROCGRAPH_CHECKARG_SIZE(3, nnz);
    ROCGRAPH_CHECKARG(3, nnz, (nnz > rows * cols), rocgraph_status_invalid_size);

    // cugraph allows setting NULL for the pointers when nnz == 0. See SWDEV_453599 for reproducer.
    // This function exists so that hipgraph can follow this behaviour without affecting rocgraph.
    ROCGRAPH_CHECKARG_ARRAY(4, nnz, csr_row_ptr);
    ROCGRAPH_CHECKARG_ARRAY(5, nnz, csr_col_ind);
    ROCGRAPH_CHECKARG_ARRAY(6, nnz, csr_val);

    ROCGRAPH_CHECKARG_ENUM(7, row_ptr_type);
    ROCGRAPH_CHECKARG_ENUM(8, col_ind_type);
    ROCGRAPH_CHECKARG_ENUM(9, idx_base);
    ROCGRAPH_CHECKARG_ENUM(10, data_type);

    *descr = new _rocgraph_spmat_descr;

    (*descr)->init = true;

    (*descr)->rows = rows;
    (*descr)->cols = cols;
    (*descr)->nnz  = nnz;

    (*descr)->row_data = csr_row_ptr;
    (*descr)->col_data = csr_col_ind;
    (*descr)->val_data = csr_val;

    (*descr)->const_row_data = csr_row_ptr;
    (*descr)->const_col_data = csr_col_ind;
    (*descr)->const_val_data = csr_val;

    (*descr)->row_type  = row_ptr_type;
    (*descr)->col_type  = col_ind_type;
    (*descr)->data_type = data_type;

    (*descr)->idx_base = idx_base;
    (*descr)->format   = rocgraph_format_csr;

    RETURN_IF_ROCGRAPH_ERROR(rocgraph_create_mat_descr(&(*descr)->descr));
    RETURN_IF_ROCGRAPH_ERROR(rocgraph_create_mat_info(&(*descr)->info));

    // Initialize descriptor
    RETURN_IF_ROCGRAPH_ERROR(rocgraph_set_mat_index_base((*descr)->descr, idx_base));

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

#ifdef __cplusplus
}
#endif
