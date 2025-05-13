/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

/*! \file
 *  \brief rocgraph-auxiliary.h provides auxilary functions in rocgraph
 */

#ifndef ROCGRAPH_AUXILIARY_H
#define ROCGRAPH_AUXILIARY_H

#include "internal/aux/rocgraph_handle_aux.h"
#include "rocgraph-export.h"
#include "sparse_utility_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
 *  \brief Return the string representation of a rocGRAPH status code enum name
 *
 *  \details
 *  \p rocgraph_get_status_name takes a rocGRAPH status as input and returns the string representation of this status.
 *  If the status is not recognized, the function returns "Unrecognized status code"
 *
 *  @param[in]
 *  status  a rocGRAPH status
 *
 *  \retval pointer to null terminated string
 */
ROCGRAPH_EXPORT
const char* rocgraph_get_status_name(rocgraph_status status);

/*! \ingroup aux_module
 *  \brief Return the rocGRAPH status code description as a string
 *
 *  \details
 *  \p rocgraph_get_status_description takes a rocGRAPH status as input and returns the status description as a string.
 *  If the status is not recognized, the function returns "Unrecognized status code"
 *
 *  @param[in]
 *  status  a rocGRAPH status
 *
 *  \retval pointer to null terminated string
 */
ROCGRAPH_EXPORT
const char* rocgraph_get_status_description(rocgraph_status status);

/*! \ingroup aux_module
 *  \brief Create a matrix descriptor
 *  \details
 *  \p rocgraph_create_mat_descr creates a matrix descriptor. It initializes
 *  \ref rocgraph_matrix_type to \ref rocgraph_matrix_type_general and
 *  \ref rocgraph_index_base to \ref rocgraph_index_base_zero. It should be destroyed
 *  at the end using rocgraph_destroy_mat_descr().
 *
 *  @param[out]
 *  descr   the pointer to the matrix descriptor.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer \p descr pointer is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_create_mat_descr(rocgraph_mat_descr* descr);

/*! \ingroup aux_module
 *  \brief Copy a matrix descriptor
 *  \details
 *  \p rocgraph_copy_mat_descr copies a matrix descriptor. Both, source and destination
 *  matrix descriptors must be initialized prior to calling \p rocgraph_copy_mat_descr.
 *
 *  @param[out]
 *  dest    the pointer to the destination matrix descriptor.
 *  @param[in]
 *  src     the pointer to the source matrix descriptor.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer \p src or \p dest pointer is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_copy_mat_descr(rocgraph_mat_descr dest, const rocgraph_mat_descr src);

/*! \ingroup aux_module
 *  \brief Destroy a matrix descriptor
 *
 *  \details
 *  \p rocgraph_destroy_mat_descr destroys a matrix descriptor and releases all
 *  resources used by the descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer \p descr is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_destroy_mat_descr(rocgraph_mat_descr descr);

/*! \ingroup aux_module
 *  \brief Specify the index base of a matrix descriptor
 *
 *  \details
 *  \p rocgraph_set_mat_index_base sets the index base of a matrix descriptor. Valid
 *  options are \ref rocgraph_index_base_zero or \ref rocgraph_index_base_one.
 *
 *  @param[inout]
 *  descr   the matrix descriptor.
 *  @param[in]
 *  base    \ref rocgraph_index_base_zero or \ref rocgraph_index_base_one.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer \p descr pointer is invalid.
 *  \retval rocgraph_status_invalid_value \p base is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_set_mat_index_base(rocgraph_mat_descr descr, rocgraph_index_base base);

/*! \ingroup aux_module
 *  \brief Get the index base of a matrix descriptor
 *
 *  \details
 *  \p rocgraph_get_mat_index_base returns the index base of a matrix descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \returns \ref rocgraph_index_base_zero or \ref rocgraph_index_base_one.
 */
ROCGRAPH_EXPORT
rocgraph_index_base rocgraph_get_mat_index_base(const rocgraph_mat_descr descr);

/*! \ingroup aux_module
 *  \brief Specify the matrix type of a matrix descriptor
 *
 *  \details
 *  \p rocgraph_set_mat_type sets the matrix type of a matrix descriptor. Valid
 *  matrix types are \ref rocgraph_matrix_type_general,
 *  \ref rocgraph_matrix_type_symmetric, \ref rocgraph_matrix_type_hermitian or
 *  \ref rocgraph_matrix_type_triangular.
 *
 *  @param[inout]
 *  descr   the matrix descriptor.
 *  @param[in]
 *  type    \ref rocgraph_matrix_type_general, \ref rocgraph_matrix_type_symmetric,
 *          \ref rocgraph_matrix_type_hermitian or
 *          \ref rocgraph_matrix_type_triangular.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer \p descr pointer is invalid.
 *  \retval rocgraph_status_invalid_value \p type is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_set_mat_type(rocgraph_mat_descr descr, rocgraph_matrix_type type);

/*! \ingroup aux_module
 *  \brief Get the matrix type of a matrix descriptor
 *
 *  \details
 *  \p rocgraph_get_mat_type returns the matrix type of a matrix descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \returns    \ref rocgraph_matrix_type_general, \ref rocgraph_matrix_type_symmetric,
 *              \ref rocgraph_matrix_type_hermitian or
 *              \ref rocgraph_matrix_type_triangular.
 */
ROCGRAPH_EXPORT
rocgraph_matrix_type rocgraph_get_mat_type(const rocgraph_mat_descr descr);

/*! \ingroup aux_module
 *  \brief Specify the matrix fill mode of a matrix descriptor
 *
 *  \details
 *  \p rocgraph_set_mat_fill_mode sets the matrix fill mode of a matrix descriptor.
 *  Valid fill modes are \ref rocgraph_fill_mode_lower or
 *  \ref rocgraph_fill_mode_upper.
 *
 *  @param[inout]
 *  descr       the matrix descriptor.
 *  @param[in]
 *  fill_mode   \ref rocgraph_fill_mode_lower or \ref rocgraph_fill_mode_upper.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer \p descr pointer is invalid.
 *  \retval rocgraph_status_invalid_value \p fill_mode is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_set_mat_fill_mode(rocgraph_mat_descr descr, rocgraph_fill_mode fill_mode);

/*! \ingroup aux_module
 *  \brief Get the matrix fill mode of a matrix descriptor
 *
 *  \details
 *  \p rocgraph_get_mat_fill_mode returns the matrix fill mode of a matrix descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \returns    \ref rocgraph_fill_mode_lower or \ref rocgraph_fill_mode_upper.
 */
ROCGRAPH_EXPORT
rocgraph_fill_mode rocgraph_get_mat_fill_mode(const rocgraph_mat_descr descr);

/*! \ingroup aux_module
 *  \brief Specify the matrix diagonal type of a matrix descriptor
 *
 *  \details
 *  \p rocgraph_set_mat_diag_type sets the matrix diagonal type of a matrix
 *  descriptor. Valid diagonal types are \ref rocgraph_diag_type_unit or
 *  \ref rocgraph_diag_type_non_unit.
 *
 *  @param[inout]
 *  descr       the matrix descriptor.
 *  @param[in]
 *  diag_type   \ref rocgraph_diag_type_unit or \ref rocgraph_diag_type_non_unit.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer \p descr pointer is invalid.
 *  \retval rocgraph_status_invalid_value \p diag_type is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_set_mat_diag_type(rocgraph_mat_descr descr, rocgraph_diag_type diag_type);

/*! \ingroup aux_module
 *  \brief Get the matrix diagonal type of a matrix descriptor
 *
 *  \details
 *  \p rocgraph_get_mat_diag_type returns the matrix diagonal type of a matrix
 *  descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \returns \ref rocgraph_diag_type_unit or \ref rocgraph_diag_type_non_unit.
 */
ROCGRAPH_EXPORT
rocgraph_diag_type rocgraph_get_mat_diag_type(const rocgraph_mat_descr descr);

/*! \ingroup aux_module
 *  \brief Specify the matrix storage mode of a matrix descriptor
 *
 *  \details
 *  \p rocgraph_set_mat_storage_mode sets the matrix storage mode of a matrix descriptor.
 *  Valid fill modes are \ref rocgraph_storage_mode_sorted or
 *  \ref rocgraph_storage_mode_unsorted.
 *
 *  @param[inout]
 *  descr           the matrix descriptor.
 *  @param[in]
 *  storage_mode    \ref rocgraph_storage_mode_sorted or
 *                  \ref rocgraph_storage_mode_unsorted.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer \p descr pointer is invalid.
 *  \retval rocgraph_status_invalid_value \p storage_mode is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_set_mat_storage_mode(rocgraph_mat_descr    descr,
                                              rocgraph_storage_mode storage_mode);

/*! \ingroup aux_module
 *  \brief Get the matrix storage mode of a matrix descriptor
 *
 *  \details
 *  \p rocgraph_get_mat_storage_mode returns the matrix storage mode of a matrix descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \returns    \ref rocgraph_storage_mode_sorted or \ref rocgraph_storage_mode_unsorted.
 */
ROCGRAPH_EXPORT
rocgraph_storage_mode rocgraph_get_mat_storage_mode(const rocgraph_mat_descr descr);

/*! \ingroup aux_module
 *  \brief Create a matrix info structure
 *
 *  \details
 *  \p rocgraph_create_mat_info creates a structure that holds the matrix info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using rocgraph_destroy_mat_info().
 *
 *  @param[inout]
 *  info    the pointer to the info structure.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer \p info pointer is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_create_mat_info(rocgraph_mat_info* info);

/*! \ingroup aux_module
 *  \brief Copy a matrix info structure
 *  \details
 *  \p rocgraph_copy_mat_info copies a matrix info structure. Both, source and destination
 *  matrix info structure must be initialized prior to calling \p rocgraph_copy_mat_info.
 *
 *  @param[out]
 *  dest    the pointer to the destination matrix info structure.
 *  @param[in]
 *  src     the pointer to the source matrix info structure.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer \p src or \p dest pointer is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_copy_mat_info(rocgraph_mat_info dest, const rocgraph_mat_info src);

/*! \ingroup aux_module
 *  \brief Destroy a matrix info structure
 *
 *  \details
 *  \p rocgraph_destroy_mat_info destroys a matrix info structure.
 *
 *  @param[in]
 *  info    the info structure.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer \p info pointer is invalid.
 *  \retval rocgraph_status_internal_error an internal error occurred.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_destroy_mat_info(rocgraph_mat_info info);

/*! \ingroup aux_module
 *  \brief Create a color info structure
 *
 *  \details
 *  \p rocgraph_create_color_info creates a structure that holds the color info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using rocgraph_destroy_color_info().
 *
 *  @param[inout]
 *  info    the pointer to the info structure.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer \p info pointer is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_create_color_info(rocgraph_color_info* info);

/*! \ingroup aux_module
 *  \brief Copy a color info structure
 *  \details
 *  \p rocgraph_copy_color_info copies a color info structure. Both, source and destination
 *  color info structure must be initialized prior to calling \p rocgraph_copy_color_info.
 *
 *  @param[out]
 *  dest    the pointer to the destination color info structure.
 *  @param[in]
 *  src     the pointer to the source color info structure.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer \p src or \p dest pointer is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_copy_color_info(rocgraph_color_info dest, const rocgraph_color_info src);

/*! \ingroup aux_module
 *  \brief Destroy a color info structure
 *
 *  \details
 *  \p rocgraph_destroy_color_info destroys a color info structure.
 *
 *  @param[in]
 *  info    the info structure.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer \p info pointer is invalid.
 *  \retval rocgraph_status_internal_error an internal error occurred.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_destroy_color_info(rocgraph_color_info info);

// Generic API

/*! \ingroup aux_module
 *  \brief Create a graph COO matrix descriptor
 *  \details
 *  \p rocgraph_create_coo_descr creates a graph COO matrix descriptor. It should be
 *  destroyed at the end using \p rocgraph_destroy_spmat_descr.
 *
 *  @param[out]
 *  descr       the pointer to the graph COO matrix descriptor.
 *  @param[in]
 *  rows        number of rows in the COO matrix.
 *  @param[in]
 *  cols        number of columns in the COO matrix
 *  @param[in]
 *  nnz         number of non-zeros in the COO matrix.
 *  @param[in]
 *  coo_row_ind row indices of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  coo_col_ind column indices of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  coo_val     values of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  idx_type    \ref rocgraph_indextype_i32 or \ref rocgraph_indextype_i64.
 *  @param[in]
 *  idx_base    \ref rocgraph_index_base_zero or \ref rocgraph_index_base_one.
 *  @param[in]
 *  data_type   \ref rocgraph_datatype_f32_r, \ref rocgraph_datatype_f64_r.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr or \p coo_row_ind or \p coo_col_ind or \p coo_val is invalid.
 *  \retval rocgraph_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocgraph_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCGRAPH_EXPORT
rocgraph_status rocgraph_create_coo_descr(rocgraph_spmat_descr* descr,
                                          int64_t               rows,
                                          int64_t               cols,
                                          int64_t               nnz,
                                          void*                 coo_row_ind,
                                          void*                 coo_col_ind,
                                          void*                 coo_val,
                                          rocgraph_indextype    idx_type,
                                          rocgraph_index_base   idx_base,
                                          rocgraph_datatype     data_type);

ROCGRAPH_EXPORT
rocgraph_status rocgraph_create_const_coo_descr(rocgraph_const_spmat_descr* descr,
                                                int64_t                     rows,
                                                int64_t                     cols,
                                                int64_t                     nnz,
                                                const void*                 coo_row_ind,
                                                const void*                 coo_col_ind,
                                                const void*                 coo_val,
                                                rocgraph_indextype          idx_type,
                                                rocgraph_index_base         idx_base,
                                                rocgraph_datatype           data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Create a graph COO AoS matrix descriptor
 *  \details
 *  \p rocgraph_create_coo_aos_descr creates a graph COO AoS matrix descriptor. It should be
 *  destroyed at the end using \p rocgraph_destroy_spmat_descr.
 *
 *  @param[out]
 *  descr       the pointer to the graph COO AoS matrix descriptor.
 *  @param[in]
 *  rows        number of rows in the COO AoS matrix.
 *  @param[in]
 *  cols        number of columns in the COO AoS matrix
 *  @param[in]
 *  nnz         number of non-zeros in the COO AoS matrix.
 *  @param[in]
 *  coo_ind     <row, column> indices of the COO AoS matrix (must be array of length \p nnz ).
 *  @param[in]
 *  coo_val     values of the COO AoS matrix (must be array of length \p nnz ).
 *  @param[in]
 *  idx_type    \ref rocgraph_indextype_i32 or \ref rocgraph_indextype_i64.
 *  @param[in]
 *  idx_base    \ref rocgraph_index_base_zero or \ref rocgraph_index_base_one.
 *  @param[in]
 *  data_type   \ref rocgraph_datatype_f32_r, \ref rocgraph_datatype_f64_r.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr or \p coo_ind or \p coo_val is invalid.
 *  \retval rocgraph_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocgraph_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_create_coo_aos_descr(rocgraph_spmat_descr* descr,
                                              int64_t               rows,
                                              int64_t               cols,
                                              int64_t               nnz,
                                              void*                 coo_ind,
                                              void*                 coo_val,
                                              rocgraph_indextype    idx_type,
                                              rocgraph_index_base   idx_base,
                                              rocgraph_datatype     data_type);

/*! \ingroup aux_module
 *  \brief Create a graph CSR matrix descriptor
 *  \details
 *  \p rocgraph_create_csr_descr creates a graph CSR matrix descriptor. It should be
 *  destroyed at the end using \p rocgraph_destroy_spmat_descr.
 *
 *  @param[out]
 *  descr        the pointer to the graph CSR matrix descriptor.
 *  @param[in]
 *  rows         number of rows in the CSR matrix.
 *  @param[in]
 *  cols         number of columns in the CSR matrix
 *  @param[in]
 *  nnz          number of non-zeros in the CSR matrix.
 *  @param[in]
 *  csr_row_ptr  row offsets of the CSR matrix (must be array of length \p rows+1 ).
 *  @param[in]
 *  csr_col_ind  column indices of the CSR matrix (must be array of length \p nnz ).
 *  @param[in]
 *  csr_val      values of the CSR matrix (must be array of length \p nnz ).
 *  @param[in]
 *  row_ptr_type \ref rocgraph_indextype_i32 or \ref rocgraph_indextype_i64.
 *  @param[in]
 *  col_ind_type \ref rocgraph_indextype_i32 or \ref rocgraph_indextype_i64.
 *  @param[in]
 *  idx_base     \ref rocgraph_index_base_zero or \ref rocgraph_index_base_one.
 *  @param[in]
 *  data_type    \ref rocgraph_datatype_f32_r, \ref rocgraph_datatype_f64_r.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr or \p csr_row_ptr or \p csr_col_ind or \p csr_val is invalid.
 *  \retval rocgraph_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocgraph_status_invalid_value if \p row_ptr_type or \p col_ind_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCGRAPH_EXPORT
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
                                          rocgraph_datatype     data_type);

ROCGRAPH_EXPORT
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
                                                rocgraph_datatype           data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Create a graph CSC matrix descriptor
 *  \details
 *  \p rocgraph_create_csc_descr creates a graph CSC matrix descriptor. It should be
 *  destroyed at the end using \p rocgraph_destroy_spmat_descr.
 *
 *  @param[out]
 *  descr       the pointer to the graph CSC matrix descriptor.
 *  @param[in]
 *  rows         number of rows in the CSC matrix.
 *  @param[in]
 *  cols         number of columns in the CSC matrix
 *  @param[in]
 *  nnz          number of non-zeros in the CSC matrix.
 *  @param[in]
 *  csc_col_ptr  column offsets of the CSC matrix (must be array of length \p cols+1 ).
 *  @param[in]
 *  csc_row_ind  row indices of the CSC matrix (must be array of length \p nnz ).
 *  @param[in]
 *  csc_val      values of the CSC matrix (must be array of length \p nnz ).
 *  @param[in]
 *  col_ptr_type \ref rocgraph_indextype_i32 or \ref rocgraph_indextype_i64.
 *  @param[in]
 *  row_ind_type \ref rocgraph_indextype_i32 or \ref rocgraph_indextype_i64.
 *  @param[in]
 *  idx_base     \ref rocgraph_index_base_zero or \ref rocgraph_index_base_one.
 *  @param[in]
 *  data_type    \ref rocgraph_datatype_f32_r, \ref rocgraph_datatype_f64_r.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr or \p csc_col_ptr or \p csc_row_ind or \p csc_val is invalid.
 *  \retval rocgraph_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocgraph_status_invalid_value if \p col_ptr_type or \p row_ind_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCGRAPH_EXPORT
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
                                          rocgraph_datatype     data_type);

ROCGRAPH_EXPORT
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
                                                rocgraph_datatype           data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Destroy a graph matrix descriptor
 *
 *  \details
 *  \p rocgraph_destroy_spmat_descr destroys a graph matrix descriptor and releases all
 *  resources used by the descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer \p descr is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_destroy_spmat_descr(rocgraph_const_spmat_descr descr);

/*! \ingroup aux_module
 *  \brief Get the fields of the graph COO matrix descriptor
 *  \details
 *  \p rocgraph_coo_get gets the fields of the graph COO matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the graph COO matrix descriptor.
 *  @param[out]
 *  rows        number of rows in the graph COO matrix.
 *  @param[out]
 *  cols        number of columns in the graph COO matrix.
 *  @param[out]
 *  nnz         number of non-zeros in graph COO matrix.
 *  @param[out]
 *  coo_row_ind row indices of the COO matrix (must be array of length \p nnz ).
 *  @param[out]
 *  coo_col_ind column indices of the COO matrix (must be array of length \p nnz ).
 *  @param[out]
 *  coo_val     values of the COO matrix (must be array of length \p nnz ).
 *  @param[out]
 *  idx_type    \ref rocgraph_indextype_i32 or \ref rocgraph_indextype_i64.
 *  @param[out]
 *  idx_base    \ref rocgraph_index_base_zero or \ref rocgraph_index_base_one.
 *  @param[out]
 *  data_type   \ref rocgraph_datatype_f32_r, \ref rocgraph_datatype_f64_r.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr or \p coo_row_ind or \p coo_col_ind or \p coo_val is invalid.
 *  \retval rocgraph_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocgraph_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCGRAPH_EXPORT
rocgraph_status rocgraph_coo_get(const rocgraph_spmat_descr descr,
                                 int64_t*                   rows,
                                 int64_t*                   cols,
                                 int64_t*                   nnz,
                                 void**                     coo_row_ind,
                                 void**                     coo_col_ind,
                                 void**                     coo_val,
                                 rocgraph_indextype*        idx_type,
                                 rocgraph_index_base*       idx_base,
                                 rocgraph_datatype*         data_type);

ROCGRAPH_EXPORT
rocgraph_status rocgraph_const_coo_get(rocgraph_const_spmat_descr descr,
                                       int64_t*                   rows,
                                       int64_t*                   cols,
                                       int64_t*                   nnz,
                                       const void**               coo_row_ind,
                                       const void**               coo_col_ind,
                                       const void**               coo_val,
                                       rocgraph_indextype*        idx_type,
                                       rocgraph_index_base*       idx_base,
                                       rocgraph_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the fields of the graph COO AoS matrix descriptor
 *  \details
 *  \p rocgraph_coo_aos_get gets the fields of the graph COO AoS matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the graph COO AoS matrix descriptor.
 *  @param[out]
 *  rows        number of rows in the graph COO AoS matrix.
 *  @param[out]
 *  cols        number of columns in the graph COO AoS matrix.
 *  @param[out]
 *  nnz         number of non-zeros in graph COO AoS matrix.
 *  @param[out]
 *  coo_ind     <row, columns> indices of the COO AoS matrix (must be array of length \p nnz ).
 *  @param[out]
 *  coo_val     values of the COO AoS matrix (must be array of length \p nnz ).
 *  @param[out]
 *  idx_type    \ref rocgraph_indextype_i32 or \ref rocgraph_indextype_i64.
 *  @param[out]
 *  idx_base    \ref rocgraph_index_base_zero or \ref rocgraph_index_base_one.
 *  @param[out]
 *  data_type   \ref rocgraph_datatype_f32_r, \ref rocgraph_datatype_f64_r.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr or \p coo_ind or \p coo_val is invalid.
 *  \retval rocgraph_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocgraph_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCGRAPH_EXPORT
rocgraph_status rocgraph_coo_aos_get(const rocgraph_spmat_descr descr,
                                     int64_t*                   rows,
                                     int64_t*                   cols,
                                     int64_t*                   nnz,
                                     void**                     coo_ind,
                                     void**                     coo_val,
                                     rocgraph_indextype*        idx_type,
                                     rocgraph_index_base*       idx_base,
                                     rocgraph_datatype*         data_type);

ROCGRAPH_EXPORT
rocgraph_status rocgraph_const_coo_aos_get(rocgraph_const_spmat_descr descr,
                                           int64_t*                   rows,
                                           int64_t*                   cols,
                                           int64_t*                   nnz,
                                           const void**               coo_ind,
                                           const void**               coo_val,
                                           rocgraph_indextype*        idx_type,
                                           rocgraph_index_base*       idx_base,
                                           rocgraph_datatype*         data_type);

/**@}*/
/*! \ingroup aux_module
 *  \brief Get the fields of the graph CSR matrix descriptor
 *  \details
 *  \p rocgraph_csr_get gets the fields of the graph CSR matrix descriptor
 *
 *  @param[in]
 *  descr        the pointer to the graph CSR matrix descriptor.
 *  @param[out]
 *  rows         number of rows in the CSR matrix.
 *  @param[out]
 *  cols         number of columns in the CSR matrix
 *  @param[out]
 *  nnz          number of non-zeros in the CSR matrix.
 *  @param[out]
 *  csr_row_ptr  row offsets of the CSR matrix (must be array of length \p rows+1 ).
 *  @param[out]
 *  csr_col_ind  column indices of the CSR matrix (must be array of length \p nnz ).
 *  @param[out]
 *  csr_val      values of the CSR matrix (must be array of length \p nnz ).
 *  @param[out]
 *  row_ptr_type \ref rocgraph_indextype_i32 or \ref rocgraph_indextype_i64.
 *  @param[out]
 *  col_ind_type \ref rocgraph_indextype_i32 or \ref rocgraph_indextype_i64.
 *  @param[out]
 *  idx_base     \ref rocgraph_index_base_zero or \ref rocgraph_index_base_one.
 *  @param[out]
 *  data_type    \ref rocgraph_datatype_f32_r, \ref rocgraph_datatype_f64_r.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr or \p csr_row_ptr or \p csr_col_ind or \p csr_val is invalid.
 *  \retval rocgraph_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocgraph_status_invalid_value if \p row_ptr_type or \p col_ind_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCGRAPH_EXPORT
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
                                 rocgraph_datatype*         data_type);

ROCGRAPH_EXPORT
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
                                       rocgraph_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the fields of the graph CSC matrix descriptor
 *  \details
 *  \p rocgraph_csc_get gets the fields of the graph CSC matrix descriptor
 *
 *  @param[in]
 *  descr        the pointer to the graph CSC matrix descriptor.
 *  @param[out]
 *  rows         number of rows in the CSC matrix.
 *  @param[out]
 *  cols         number of columns in the CSC matrix
 *  @param[out]
 *  nnz          number of non-zeros in the CSC matrix.
 *  @param[out]
 *  csc_col_ptr  column offsets of the CSC matrix (must be array of length \p cols+1 ).
 *  @param[out]
 *  csc_row_ind  row indices of the CSC matrix (must be array of length \p nnz ).
 *  @param[out]
 *  csc_val      values of the CSC matrix (must be array of length \p nnz ).
 *  @param[out]
 *  col_ptr_type \ref rocgraph_indextype_i32 or \ref rocgraph_indextype_i64.
 *  @param[out]
 *  row_ind_type \ref rocgraph_indextype_i32 or \ref rocgraph_indextype_i64.
 *  @param[out]
 *  idx_base     \ref rocgraph_index_base_zero or \ref rocgraph_index_base_one.
 *  @param[out]
 *  data_type    \ref rocgraph_datatype_f32_r, \ref rocgraph_datatype_f64_r.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr or \p csc_col_ptr or \p csc_row_ind or \p csr_val is invalid.
 *  \retval rocgraph_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocgraph_status_invalid_value if \p col_ptr_type or \p row_ind_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCGRAPH_EXPORT
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
                                 rocgraph_datatype*         data_type);

ROCGRAPH_EXPORT
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
                                       rocgraph_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Set the row indices, column indices and values array in the graph COO matrix descriptor
 *
 *  @param[inout]
 *  descr   the pointer to the graph vector descriptor.
 *  @param[in]
 *  coo_row_ind row indices of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  coo_col_ind column indices of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  coo_val     values of the COO matrix (must be array of length \p nnz ).
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr or \p coo_row_ind or \p coo_col_ind or \p coo_val is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_coo_set_pointers(rocgraph_spmat_descr descr,
                                          void*                coo_row_ind,
                                          void*                coo_col_ind,
                                          void*                coo_val);

/*! \ingroup aux_module
 *  \brief Set the <row, column> indices and values array in the graph COO AoS matrix descriptor
 *
 *  @param[inout]
 *  descr   the pointer to the graph vector descriptor.
 *  @param[in]
 *  coo_ind <row, column> indices of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  coo_val values of the COO matrix (must be array of length \p nnz ).
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr or \p coo_ind or \p coo_val is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status
    rocgraph_coo_aos_set_pointers(rocgraph_spmat_descr descr, void* coo_ind, void* coo_val);

/*! \ingroup aux_module
 *  \brief Set the row offsets, column indices and values array in the graph CSR matrix descriptor
 *
 *  @param[inout]
 *  descr   the pointer to the graph vector descriptor.
 *  @param[in]
 *  csr_row_ptr  row offsets of the CSR matrix (must be array of length \p rows+1 ).
 *  @param[in]
 *  csr_col_ind  column indices of the CSR matrix (must be array of length \p nnz ).
 *  @param[in]
 *  csr_val      values of the CSR matrix (must be array of length \p nnz ).
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr or \p coo_ind or \p coo_val is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_csr_set_pointers(rocgraph_spmat_descr descr,
                                          void*                csr_row_ptr,
                                          void*                csr_col_ind,
                                          void*                csr_val);

/*! \ingroup aux_module
 *  \brief Set the column offsets, row indices and values array in the graph CSC matrix descriptor
 *
 *  @param[inout]
 *  descr       the pointer to the graph vector descriptor.
 *  @param[in]
 *  csc_col_ptr column offsets of the CSC matrix (must be array of length \p cols+1 ).
 *  @param[in]
 *  csc_row_ind row indices of the CSC matrix (must be array of length \p nnz ).
 *  @param[in]
 *  csc_val     values of the CSC matrix (must be array of length \p nnz ).
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr or \p csc_col_ptr or \p csc_row_ind or \p csc_val is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_csc_set_pointers(rocgraph_spmat_descr descr,
                                          void*                csc_col_ptr,
                                          void*                csc_row_ind,
                                          void*                csc_val);

/*! \ingroup aux_module
 *  \brief Get the number of rows, columns and non-zeros from the graph matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the graph matrix descriptor.
 *  @param[out]
 *  rows        number of rows in the graph matrix.
 *  @param[out]
 *  cols        number of columns in the graph matrix.
 *  @param[out]
 *  nnz         number of non-zeros in graph matrix.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr is invalid.
 *  \retval rocgraph_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_spmat_get_size(rocgraph_const_spmat_descr descr,
                                        int64_t*                   rows,
                                        int64_t*                   cols,
                                        int64_t*                   nnz);

/*! \ingroup aux_module
 *  \brief Get the graph matrix format from the graph matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the graph matrix descriptor.
 *  @param[out]
 *  format      \ref rocgraph_format_coo or \ref rocgraph_format_coo_aos or
 *              \ref rocgraph_format_csr or \ref rocgraph_format_csc or
 *              \ref rocgraph_format_ell
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr is invalid.
 *  \retval rocgraph_status_invalid_value if \p format is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_spmat_get_format(rocgraph_const_spmat_descr descr,
                                          rocgraph_format*           format);

/*! \ingroup aux_module
 *  \brief Get the graph matrix index base from the graph matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the graph matrix descriptor.
 *  @param[out]
 *  idx_base    \ref rocgraph_index_base_zero or \ref rocgraph_index_base_one
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr is invalid.
 *  \retval rocgraph_status_invalid_value if \p idx_base is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_spmat_get_index_base(rocgraph_const_spmat_descr descr,
                                              rocgraph_index_base*       idx_base);

/*! \ingroup aux_module
 *  \brief Get the values array from the graph matrix descriptor
 *
 *  @param[in]
 *  descr     the pointer to the graph matrix descriptor.
 *  @param[out]
 *  values    values array of the graph matrix.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr or \p values is invalid.
 */
/**@{*/
ROCGRAPH_EXPORT
rocgraph_status rocgraph_spmat_get_values(rocgraph_spmat_descr descr, void** values);

ROCGRAPH_EXPORT
rocgraph_status rocgraph_const_spmat_get_values(rocgraph_const_spmat_descr descr,
                                                const void**               values);
/**@}*/

/*! \ingroup aux_module
 *  \brief Set the values array in the graph matrix descriptor
 *
 *  @param[inout]
 *  descr     the pointer to the graph matrix descriptor.
 *  @param[in]
 *  values    values array of the graph matrix.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr or \p values is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_spmat_set_values(rocgraph_spmat_descr descr, void* values);

/*! \ingroup aux_module
 *  \brief Get the strided batch count from the graph matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the graph matrix descriptor.
 *  @param[out]
 *  batch_count batch_count of the graph matrix.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr is invalid.
 *  \retval rocgraph_status_invalid_size if \p batch_count is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_spmat_get_strided_batch(rocgraph_const_spmat_descr descr,
                                                 rocgraph_int*              batch_count);

/*! \ingroup aux_module
 *  \brief Set the strided batch count in the graph matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the graph matrix descriptor.
 *  @param[in]
 *  batch_count batch_count of the graph matrix.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr is invalid.
 *  \retval rocgraph_status_invalid_size if \p batch_count is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_spmat_set_strided_batch(rocgraph_spmat_descr descr,
                                                 rocgraph_int         batch_count);

/*! \ingroup aux_module
 *  \brief Set the batch count and batch stride in the graph COO matrix descriptor
 *
 *  @param[inout]
 *  descr        the pointer to the graph COO matrix descriptor.
 *  @param[in]
 *  batch_count  batch_count of the graph COO matrix.
 *  @param[in]
 *  batch_stride batch stride of the graph COO matrix.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr is invalid.
 *  \retval rocgraph_status_invalid_size if \p batch_count or \p batch_stride is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_coo_set_strided_batch(rocgraph_spmat_descr descr,
                                               rocgraph_int         batch_count,
                                               int64_t              batch_stride);

/*! \ingroup aux_module
 *  \brief Set the batch count, row offset batch stride and the column indices batch stride in the graph CSR matrix descriptor
 *
 *  @param[inout]
 *  descr                       the pointer to the graph CSR matrix descriptor.
 *  @param[in]
 *  batch_count                 batch_count of the graph CSR matrix.
 *  @param[in]
 *  offsets_batch_stride        row offset batch stride of the graph CSR matrix.
 *  @param[in]
 *  columns_values_batch_stride column indices batch stride of the graph CSR matrix.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr is invalid.
 *  \retval rocgraph_status_invalid_size if \p batch_count or \p offsets_batch_stride or \p columns_values_batch_stride is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_csr_set_strided_batch(rocgraph_spmat_descr descr,
                                               rocgraph_int         batch_count,
                                               int64_t              offsets_batch_stride,
                                               int64_t              columns_values_batch_stride);

/*! \ingroup aux_module
 *  \brief Set the batch count, column offset batch stride and the row indices batch stride in the graph CSC matrix descriptor
 *
 *  @param[inout]
 *  descr                       the pointer to the graph CSC matrix descriptor.
 *  @param[in]
 *  batch_count                 batch_count of the graph CSC matrix.
 *  @param[in]
 *  offsets_batch_stride        column offset batch stride of the graph CSC matrix.
 *  @param[in]
 *  rows_values_batch_stride    row indices batch stride of the graph CSC matrix.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr is invalid.
 *  \retval rocgraph_status_invalid_size if \p batch_count or \p offsets_batch_stride or \p rows_values_batch_stride is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_csc_set_strided_batch(rocgraph_spmat_descr descr,
                                               rocgraph_int         batch_count,
                                               int64_t              offsets_batch_stride,
                                               int64_t              rows_values_batch_stride);

/*! \ingroup aux_module
 *  \brief Get the requested attribute data from the graph matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the graph matrix descriptor.
 *  @param[in]
 *  attribute \ref rocgraph_spmat_fill_mode or \ref rocgraph_spmat_diag_type or
 *            \ref rocgraph_spmat_matrix_type or \ref rocgraph_spmat_storage_mode
 *  @param[out]
 *  data      attribute data
 *  @param[in]
 *  data_size attribute data size.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr or \p data is invalid.
 *  \retval rocgraph_status_invalid_value if \p attribute is invalid.
 *  \retval rocgraph_status_invalid_size if \p data_size is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_spmat_get_attribute(rocgraph_const_spmat_descr descr,
                                             rocgraph_spmat_attribute   attribute,
                                             void*                      data,
                                             size_t                     data_size);

/*! \ingroup aux_module
 *  \brief Set the requested attribute data in the graph matrix descriptor
 *
 *  @param[inout]
 *  descr       the pointer to the graph matrix descriptor.
 *  @param[in]
 *  attribute \ref rocgraph_spmat_fill_mode or \ref rocgraph_spmat_diag_type or
 *            \ref rocgraph_spmat_matrix_type or \ref rocgraph_spmat_storage_mode
 *  @param[in]
 *  data      attribute data
 *  @param[in]
 *  data_size attribute data size.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr or \p data is invalid.
 *  \retval rocgraph_status_invalid_value if \p attribute is invalid.
 *  \retval rocgraph_status_invalid_size if \p data_size is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_spmat_set_attribute(rocgraph_spmat_descr     descr,
                                             rocgraph_spmat_attribute attribute,
                                             const void*              data,
                                             size_t                   data_size);

/*! \ingroup aux_module
 *  \brief Create a dense vector descriptor
 *  \details
 *  \p rocgraph_create_dnvec_descr creates a dense vector descriptor. It should be
 *  destroyed at the end using rocgraph_destroy_dnvec_descr().
 *
 *  @param[out]
 *  descr   the pointer to the dense vector descriptor.
 *  @param[in]
 *  size   size of the dense vector.
 *  @param[in]
 *  values   non-zero values in the dense vector (must be array of length \p size ).
 *  @param[in]
 *  data_type   \ref rocgraph_datatype_f32_r, \ref rocgraph_datatype_f64_r
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr or \p values is invalid.
 *  \retval rocgraph_status_invalid_size if \p size is invalid.
 *  \retval rocgraph_status_invalid_value if \p data_type is invalid.
 */
/**@{*/
ROCGRAPH_EXPORT
rocgraph_status rocgraph_create_dnvec_descr(rocgraph_dnvec_descr* descr,
                                            int64_t               size,
                                            void*                 values,
                                            rocgraph_datatype     data_type);

ROCGRAPH_EXPORT
rocgraph_status rocgraph_create_const_dnvec_descr(rocgraph_const_dnvec_descr* descr,
                                                  int64_t                     size,
                                                  const void*                 values,
                                                  rocgraph_datatype           data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Destroy a dense vector descriptor
 *
 *  \details
 *  \p rocgraph_destroy_dnvec_descr destroys a dense vector descriptor and releases all
 *  resources used by the descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer \p descr is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_destroy_dnvec_descr(rocgraph_const_dnvec_descr descr);

/*! \ingroup aux_module
 *  \brief Get the fields of the dense vector descriptor
 *  \details
 *  \p rocgraph_dnvec_get gets the fields of the dense vector descriptor
 *
 *  @param[in]
 *  descr   the pointer to the dense vector descriptor.
 *  @param[out]
 *  size   size of the dense vector.
 *  @param[out]
 *  values   non-zero values in the dense vector (must be array of length \p size ).
 *  @param[out]
 *  data_type   \ref rocgraph_datatype_f32_r, \ref rocgraph_datatype_f64_r.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr or \p values is invalid.
 *  \retval rocgraph_status_invalid_size if \p size is invalid.
 *  \retval rocgraph_status_invalid_value if \p data_type is invalid.
 */
/**@{*/
ROCGRAPH_EXPORT
rocgraph_status rocgraph_dnvec_get(const rocgraph_dnvec_descr descr,
                                   int64_t*                   size,
                                   void**                     values,
                                   rocgraph_datatype*         data_type);

ROCGRAPH_EXPORT
rocgraph_status rocgraph_const_dnvec_get(rocgraph_const_dnvec_descr descr,
                                         int64_t*                   size,
                                         const void**               values,
                                         rocgraph_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the values array from a dense vector descriptor
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *  @param[out]
 *  values   non-zero values in the dense vector (must be array of length \p size ).
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer \p descr or \p values is invalid.
 */
/**@{*/
ROCGRAPH_EXPORT
rocgraph_status rocgraph_dnvec_get_values(const rocgraph_dnvec_descr descr, void** values);

ROCGRAPH_EXPORT
rocgraph_status rocgraph_const_dnvec_get_values(rocgraph_const_dnvec_descr descr,
                                                const void**               values);
/**@}*/

/*! \ingroup aux_module
 *  \brief Set the values array in a dense vector descriptor
 *
 *  @param[inout]
 *  descr   the matrix descriptor.
 *  @param[in]
 *  values   non-zero values in the dense vector (must be array of length \p size ).
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer \p descr or \p values is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_dnvec_set_values(rocgraph_dnvec_descr descr, void* values);

/*! \ingroup aux_module
 *  \brief Create a dense matrix descriptor
 *  \details
 *  \p rocgraph_create_dnmat_descr creates a dense matrix descriptor. It should be
 *  destroyed at the end using rocgraph_destroy_dnmat_descr().
 *
 *  @param[out]
 *  descr     the pointer to the dense matrix descriptor.
 *  @param[in]
 *  rows      number of rows in the dense matrix.
 *  @param[in]
 *  cols      number of columns in the dense matrix.
 *  @param[in]
 *  ld        leading dimension of the dense matrix.
 *  @param[in]
 *  values    non-zero values in the dense vector (must be array of length
 *            \p ld*rows if \p order=rocgraph_order_column or \p ld*cols if \p order=rocgraph_order_row ).
 *  @param[in]
 *  data_type \ref rocgraph_datatype_f32_r, \ref rocgraph_datatype_f64_r.
 *  @param[in]
 *  order     \ref rocgraph_order_row or \ref rocgraph_order_column.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr or \p values is invalid.
 *  \retval rocgraph_status_invalid_size if \p rows or \p cols or \p ld is invalid.
 *  \retval rocgraph_status_invalid_value if \p data_type or \p order is invalid.
 */
/**@{*/
ROCGRAPH_EXPORT
rocgraph_status rocgraph_create_dnmat_descr(rocgraph_dnmat_descr* descr,
                                            int64_t               rows,
                                            int64_t               cols,
                                            int64_t               ld,
                                            void*                 values,
                                            rocgraph_datatype     data_type,
                                            rocgraph_order        order);

ROCGRAPH_EXPORT
rocgraph_status rocgraph_create_const_dnmat_descr(rocgraph_const_dnmat_descr* descr,
                                                  int64_t                     rows,
                                                  int64_t                     cols,
                                                  int64_t                     ld,
                                                  const void*                 values,
                                                  rocgraph_datatype           data_type,
                                                  rocgraph_order              order);
/**@}*/

/*! \ingroup aux_module
 *  \brief Destroy a dense matrix descriptor
 *
 *  \details
 *  \p rocgraph_destroy_dnmat_descr destroys a dense matrix descriptor and releases all
 *  resources used by the descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer \p descr is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_destroy_dnmat_descr(rocgraph_const_dnmat_descr descr);

/*! \ingroup aux_module
 *  \brief Get the fields of the dense matrix descriptor
 *
 *  @param[in]
 *  descr   the pointer to the dense matrix descriptor.
 *  @param[out]
 *  rows   number of rows in the dense matrix.
 *  @param[out]
 *  cols   number of columns in the dense matrix.
 *  @param[out]
 *  ld        leading dimension of the dense matrix.
 *  @param[out]
 *  values    non-zero values in the dense matrix (must be array of length
 *            \p ld*rows if \p order=rocgraph_order_column or \p ld*cols if \p order=rocgraph_order_row ).
 *  @param[out]
 *  data_type   \ref rocgraph_datatype_f32_r, \ref rocgraph_datatype_f64_r.
 *  @param[out]
 *  order     \ref rocgraph_order_row or \ref rocgraph_order_column.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr or \p values is invalid.
 *  \retval rocgraph_status_invalid_size if \p rows or \p cols or \p ld is invalid.
 *  \retval rocgraph_status_invalid_value if \p data_type or \p order is invalid.
 */
/**@{*/
ROCGRAPH_EXPORT
rocgraph_status rocgraph_dnmat_get(const rocgraph_dnmat_descr descr,
                                   int64_t*                   rows,
                                   int64_t*                   cols,
                                   int64_t*                   ld,
                                   void**                     values,
                                   rocgraph_datatype*         data_type,
                                   rocgraph_order*            order);

ROCGRAPH_EXPORT
rocgraph_status rocgraph_const_dnmat_get(rocgraph_const_dnmat_descr descr,
                                         int64_t*                   rows,
                                         int64_t*                   cols,
                                         int64_t*                   ld,
                                         const void**               values,
                                         rocgraph_datatype*         data_type,
                                         rocgraph_order*            order);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the values array from the dense matrix descriptor
 *
 *  @param[in]
 *  descr   the pointer to the dense matrix descriptor.
 *  @param[out]
 *  values    non-zero values in the dense matrix (must be array of length
 *            \p ld*rows if \p order=rocgraph_order_column or \p ld*cols if \p order=rocgraph_order_row ).
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr or \p values is invalid.
 */
/**@{*/
ROCGRAPH_EXPORT
rocgraph_status rocgraph_dnmat_get_values(const rocgraph_dnmat_descr descr, void** values);

ROCGRAPH_EXPORT
rocgraph_status rocgraph_const_dnmat_get_values(rocgraph_const_dnmat_descr descr,
                                                const void**               values);
/**@}*/

/*! \ingroup aux_module
 *  \brief Set the values array in a dense matrix descriptor
 *
 *  @param[inout]
 *  descr   the matrix descriptor.
 *  @param[in]
 *  values    non-zero values in the dense matrix (must be array of length
 *            \p ld*rows if \p order=rocgraph_order_column or \p ld*cols if \p order=rocgraph_order_row ).
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer \p descr or \p values is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_dnmat_set_values(rocgraph_dnmat_descr descr, void* values);

/*! \ingroup aux_module
 *  \brief Get the batch count and batch stride from the dense matrix descriptor
 *
 *  @param[in]
 *  descr        the pointer to the dense matrix descriptor.
 *  @param[out]
 *  batch_count  the batch count in the dense matrix.
 *  @param[out]
 *  batch_stride the batch stride in the dense matrix.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr is invalid.
 *  \retval rocgraph_status_invalid_size if \p batch_count or \p batch_stride is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_dnmat_get_strided_batch(rocgraph_const_dnmat_descr descr,
                                                 rocgraph_int*              batch_count,
                                                 int64_t*                   batch_stride);

/*! \ingroup aux_module
 *  \brief Set the batch count and batch stride in the dense matrix descriptor
 *
 *  @param[inout]
 *  descr        the pointer to the dense matrix descriptor.
 *  @param[in]
 *  batch_count  the batch count in the dense matrix.
 *  @param[in]
 *  batch_stride the batch stride in the dense matrix.
 *
 *  \retval rocgraph_status_success the operation completed successfully.
 *  \retval rocgraph_status_invalid_pointer if \p descr is invalid.
 *  \retval rocgraph_status_invalid_size if \p batch_count or \p batch_stride is invalid.
 */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_dnmat_set_strided_batch(rocgraph_dnmat_descr descr,
                                                 rocgraph_int         batch_count,
                                                 int64_t              batch_stride);

#ifdef __cplusplus
}
#endif

#endif /* ROCGRAPH_AUXILIARY_H */
