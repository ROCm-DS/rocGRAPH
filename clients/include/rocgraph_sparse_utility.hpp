/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#include "rocgraph/internal/types/rocgraph_error_t.h"
#include "rocgraph/internal/types/rocgraph_handle_t.h"
#include "rocgraph/internal/types/rocgraph_status.h"
#ifdef __cplusplus
extern "C" {
#endif

#if defined(rocgraph_ILP64)
typedef int64_t rocgraph_int;
#else
typedef int32_t rocgraph_int;
#endif

typedef struct ihipStream_t* hipStream_t;

typedef struct _rocgraph_mat_descr* rocgraph_mat_descr;

typedef struct _rocgraph_hyb_mat* rocgraph_hyb_mat;

typedef struct _rocgraph_mat_info* rocgraph_mat_info;

typedef struct _rocgraph_spvec_descr* rocgraph_spvec_descr;

typedef struct _rocgraph_spvec_descr const* rocgraph_const_spvec_descr;

typedef struct _rocgraph_spmat_descr* rocgraph_spmat_descr;

typedef struct _rocgraph_spmat_descr const* rocgraph_const_spmat_descr;

typedef struct _rocgraph_dnvec_descr* rocgraph_dnvec_descr;

typedef struct _rocgraph_dnvec_descr const* rocgraph_const_dnvec_descr;

typedef struct _rocgraph_dnmat_descr* rocgraph_dnmat_descr;

typedef struct _rocgraph_dnmat_descr const* rocgraph_const_dnmat_descr;

typedef struct _rocgraph_color_info* rocgraph_color_info;

typedef enum rocgraph_operation_
{
    rocgraph_operation_none                = 111,
    rocgraph_operation_transpose           = 112,
    rocgraph_operation_conjugate_transpose = 113
} rocgraph_operation;

typedef enum rocgraph_index_base_
{
    rocgraph_index_base_zero = 0,
    rocgraph_index_base_one  = 1
} rocgraph_index_base;

typedef enum rocgraph_matrix_type_
{
    rocgraph_matrix_type_general    = 0,
    rocgraph_matrix_type_symmetric  = 1,
    rocgraph_matrix_type_hermitian  = 2,
    rocgraph_matrix_type_triangular = 3
} rocgraph_matrix_type;

typedef enum rocgraph_diag_type_
{
    rocgraph_diag_type_non_unit = 0,
    rocgraph_diag_type_unit     = 1
} rocgraph_diag_type;

typedef enum rocgraph_fill_mode_
{
    rocgraph_fill_mode_lower = 0,
    rocgraph_fill_mode_upper = 1
} rocgraph_fill_mode;

typedef enum rocgraph_storage_mode_
{
    rocgraph_storage_mode_sorted   = 0,
    rocgraph_storage_mode_unsorted = 1
} rocgraph_storage_mode;

typedef enum rocgraph_action_
{
    rocgraph_action_symbolic = 0,
    rocgraph_action_numeric  = 1
} rocgraph_action;

typedef enum rocgraph_direction_
{
    rocgraph_direction_row    = 0,
    rocgraph_direction_column = 1
} rocgraph_direction;

typedef enum rocgraph_hyb_partition_
{
    rocgraph_hyb_partition_auto = 0,
    rocgraph_hyb_partition_user = 1,
    rocgraph_hyb_partition_max  = 2
} rocgraph_hyb_partition;

typedef enum rocgraph_analysis_policy_
{
    rocgraph_analysis_policy_reuse = 0,
    rocgraph_analysis_policy_force = 1
} rocgraph_analysis_policy;

typedef enum rocgraph_solve_policy_
{
    rocgraph_solve_policy_auto = 0
} rocgraph_solve_policy;

typedef enum rocgraph_layer_mode
{
    rocgraph_layer_mode_none      = 0x0,
    rocgraph_layer_mode_log_trace = 0x1,
    rocgraph_layer_mode_log_bench = 0x2,
    rocgraph_layer_mode_log_debug = 0x4
} rocgraph_layer_mode;

typedef enum rocgraph_data_status_
{
    rocgraph_data_status_success            = 0,
    rocgraph_data_status_inf                = 1,
    rocgraph_data_status_nan                = 2,
    rocgraph_data_status_invalid_offset_ptr = 3,
    rocgraph_data_status_invalid_index      = 4,
    rocgraph_data_status_duplicate_entry    = 5,
    rocgraph_data_status_invalid_sorting    = 6,
    rocgraph_data_status_invalid_fill       = 7
} rocgraph_data_status;

typedef enum rocgraph_indextype_
{
    rocgraph_indextype_u16 = 1,
    rocgraph_indextype_i32 = 2,
    rocgraph_indextype_i64 = 3
} rocgraph_indextype;

typedef enum rocgraph_datatype_
{
    rocgraph_datatype_f32_r = 151,
    rocgraph_datatype_f64_r = 152,
    rocgraph_datatype_i8_r  = 160,
    rocgraph_datatype_u8_r  = 161,
    rocgraph_datatype_i32_r = 162,
    rocgraph_datatype_u32_r = 163
} rocgraph_datatype;

typedef enum rocgraph_format_
{
    rocgraph_format_coo     = 0,
    rocgraph_format_coo_aos = 1,
    rocgraph_format_csr     = 2,
    rocgraph_format_csc     = 3,
    rocgraph_format_ell     = 4,
    rocgraph_format_bell    = 5,
    rocgraph_format_bsr     = 6
} rocgraph_format;

typedef enum rocgraph_order_
{
    rocgraph_order_row    = 0,
    rocgraph_order_column = 1
} rocgraph_order;

typedef enum rocgraph_spmat_attribute_
{
    rocgraph_spmat_fill_mode    = 0,
    rocgraph_spmat_diag_type    = 1,
    rocgraph_spmat_matrix_type  = 2,
    rocgraph_spmat_storage_mode = 3
} rocgraph_spmat_attribute;

typedef enum rocgraph_spmv_stage_
{
    rocgraph_spmv_stage_buffer_size = 1,
    rocgraph_spmv_stage_preprocess  = 2,
    rocgraph_spmv_stage_compute     = 3
} rocgraph_spmv_stage;

typedef enum rocgraph_spmv_alg_
{
    rocgraph_spmv_alg_default      = 0,
    rocgraph_spmv_alg_coo          = 1,
    rocgraph_spmv_alg_csr_adaptive = 2,
    rocgraph_spmv_alg_csr_stream   = 3,
    rocgraph_spmv_alg_ell          = 4,
    rocgraph_spmv_alg_coo_atomic   = 5,
    rocgraph_spmv_alg_bsr          = 6,
    rocgraph_spmv_alg_csr_lrb      = 7
} rocgraph_spmv_alg;

typedef enum rocgraph_spmm_alg_
{
    rocgraph_spmm_alg_default = 0,
    rocgraph_spmm_alg_csr,
    rocgraph_spmm_alg_coo_segmented,
    rocgraph_spmm_alg_coo_atomic,
    rocgraph_spmm_alg_csr_row_split,
    rocgraph_spmm_alg_csr_merge,
    rocgraph_spmm_alg_coo_segmented_atomic,
    rocgraph_spmm_alg_csr_merge_path,
    rocgraph_spmm_alg_csr_nnz_split = rocgraph_spmm_alg_csr_merge
} rocgraph_spmm_alg;

typedef enum rocgraph_sddmm_alg_
{
    rocgraph_sddmm_alg_default = 0,
    rocgraph_sddmm_alg_dense   = 1
} rocgraph_sddmm_alg;

typedef enum rocgraph_graph_to_dense_alg_
{
    rocgraph_graph_to_dense_alg_default = 0,
} rocgraph_graph_to_dense_alg;

typedef enum rocgraph_dense_to_graph_alg_
{
    rocgraph_dense_to_graph_alg_default = 0,
} rocgraph_dense_to_graph_alg;

typedef enum rocgraph_spmm_stage_
{
    rocgraph_spmm_stage_buffer_size = 1,
    rocgraph_spmm_stage_preprocess  = 2,
    rocgraph_spmm_stage_compute     = 3
} rocgraph_spmm_stage;

typedef enum rocgraph_spgemm_stage_
{
    rocgraph_spgemm_stage_buffer_size = 1,
    rocgraph_spgemm_stage_nnz         = 2,
    rocgraph_spgemm_stage_compute     = 3,
    rocgraph_spgemm_stage_symbolic    = 4,
    rocgraph_spgemm_stage_numeric     = 5
} rocgraph_spgemm_stage;

typedef enum rocgraph_spgemm_alg_
{
    rocgraph_spgemm_alg_default = 0
} rocgraph_spgemm_alg;

const char* rocgraph_get_status_name(rocgraph_status status);

const char* rocgraph_get_status_description(rocgraph_status status);

rocgraph_status rocgraph_create_mat_descr(rocgraph_mat_descr* descr);

rocgraph_status rocgraph_copy_mat_descr(rocgraph_mat_descr dest, const rocgraph_mat_descr src);

rocgraph_status rocgraph_destroy_mat_descr(rocgraph_mat_descr descr);

rocgraph_status rocgraph_set_mat_index_base(rocgraph_mat_descr descr, rocgraph_index_base base);

rocgraph_index_base rocgraph_get_mat_index_base(const rocgraph_mat_descr descr);

rocgraph_status rocgraph_set_mat_type(rocgraph_mat_descr descr, rocgraph_matrix_type type);

rocgraph_matrix_type rocgraph_get_mat_type(const rocgraph_mat_descr descr);

rocgraph_status rocgraph_set_mat_fill_mode(rocgraph_mat_descr descr, rocgraph_fill_mode fill_mode);

rocgraph_fill_mode rocgraph_get_mat_fill_mode(const rocgraph_mat_descr descr);

rocgraph_status rocgraph_set_mat_diag_type(rocgraph_mat_descr descr, rocgraph_diag_type diag_type);

rocgraph_diag_type rocgraph_get_mat_diag_type(const rocgraph_mat_descr descr);

rocgraph_status rocgraph_set_mat_storage_mode(rocgraph_mat_descr    descr,
                                              rocgraph_storage_mode storage_mode);

rocgraph_storage_mode rocgraph_get_mat_storage_mode(const rocgraph_mat_descr descr);

rocgraph_status rocgraph_create_hyb_mat(rocgraph_hyb_mat* hyb);

rocgraph_status rocgraph_copy_hyb_mat(rocgraph_hyb_mat dest, const rocgraph_hyb_mat src);

rocgraph_status rocgraph_destroy_hyb_mat(rocgraph_hyb_mat hyb);

rocgraph_status rocgraph_create_mat_info(rocgraph_mat_info* info);

rocgraph_status rocgraph_copy_mat_info(rocgraph_mat_info dest, const rocgraph_mat_info src);

rocgraph_status rocgraph_destroy_mat_info(rocgraph_mat_info info);

rocgraph_status rocgraph_create_color_info(rocgraph_color_info* info);

rocgraph_status rocgraph_copy_color_info(rocgraph_color_info dest, const rocgraph_color_info src);

rocgraph_status rocgraph_destroy_color_info(rocgraph_color_info info);

rocgraph_status rocgraph_create_spvec_descr(rocgraph_spvec_descr* descr,
                                            int64_t               size,
                                            int64_t               nnz,
                                            void*                 indices,
                                            void*                 values,
                                            rocgraph_indextype    idx_type,
                                            rocgraph_index_base   idx_base,
                                            rocgraph_datatype     data_type);

rocgraph_status rocgraph_create_const_spvec_descr(rocgraph_const_spvec_descr* descr,
                                                  int64_t                     size,
                                                  int64_t                     nnz,
                                                  const void*                 indices,
                                                  const void*                 values,
                                                  rocgraph_indextype          idx_type,
                                                  rocgraph_index_base         idx_base,
                                                  rocgraph_datatype           data_type);

rocgraph_status rocgraph_destroy_spvec_descr(rocgraph_const_spvec_descr descr);

rocgraph_status rocgraph_spvec_get(const rocgraph_spvec_descr descr,
                                   int64_t*                   size,
                                   int64_t*                   nnz,
                                   void**                     indices,
                                   void**                     values,
                                   rocgraph_indextype*        idx_type,
                                   rocgraph_index_base*       idx_base,
                                   rocgraph_datatype*         data_type);

rocgraph_status rocgraph_const_spvec_get(rocgraph_const_spvec_descr descr,
                                         int64_t*                   size,
                                         int64_t*                   nnz,
                                         const void**               indices,
                                         const void**               values,
                                         rocgraph_indextype*        idx_type,
                                         rocgraph_index_base*       idx_base,
                                         rocgraph_datatype*         data_type);

rocgraph_status rocgraph_spvec_get_index_base(rocgraph_const_spvec_descr descr,
                                              rocgraph_index_base*       idx_base);

rocgraph_status rocgraph_spvec_get_values(const rocgraph_spvec_descr descr, void** values);

rocgraph_status rocgraph_const_spvec_get_values(rocgraph_const_spvec_descr descr,
                                                const void**               values);

rocgraph_status rocgraph_spvec_set_values(rocgraph_spvec_descr descr, void* values);

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

rocgraph_status rocgraph_create_coo_aos_descr(rocgraph_spmat_descr* descr,
                                              int64_t               rows,
                                              int64_t               cols,
                                              int64_t               nnz,
                                              void*                 coo_ind,
                                              void*                 coo_val,
                                              rocgraph_indextype    idx_type,
                                              rocgraph_index_base   idx_base,
                                              rocgraph_datatype     data_type);

rocgraph_status rocgraph_create_bsr_descr(rocgraph_spmat_descr* descr,
                                          int64_t               mb,
                                          int64_t               nb,
                                          int64_t               nnzb,
                                          rocgraph_direction    block_dir,
                                          int64_t               block_dim,
                                          void*                 bsr_row_ptr,
                                          void*                 bsr_col_ind,
                                          void*                 bsr_val,
                                          rocgraph_indextype    row_ptr_type,
                                          rocgraph_indextype    col_ind_type,
                                          rocgraph_index_base   idx_base,
                                          rocgraph_datatype     data_type);

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

rocgraph_status rocgraph_create_ell_descr(rocgraph_spmat_descr* descr,
                                          int64_t               rows,
                                          int64_t               cols,
                                          void*                 ell_col_ind,
                                          void*                 ell_val,
                                          int64_t               ell_width,
                                          rocgraph_indextype    idx_type,
                                          rocgraph_index_base   idx_base,
                                          rocgraph_datatype     data_type);

rocgraph_status rocgraph_create_bell_descr(rocgraph_spmat_descr* descr,
                                           int64_t               rows,
                                           int64_t               cols,
                                           rocgraph_direction    ell_block_dir,
                                           int64_t               ell_block_dim,
                                           int64_t               ell_cols,
                                           void*                 ell_col_ind,
                                           void*                 ell_val,
                                           rocgraph_indextype    idx_type,
                                           rocgraph_index_base   idx_base,
                                           rocgraph_datatype     data_type);

rocgraph_status rocgraph_create_const_bell_descr(rocgraph_const_spmat_descr* descr,
                                                 int64_t                     rows,
                                                 int64_t                     cols,
                                                 rocgraph_direction          ell_block_dir,
                                                 int64_t                     ell_block_dim,
                                                 int64_t                     ell_cols,
                                                 const void*                 ell_col_ind,
                                                 const void*                 ell_val,
                                                 rocgraph_indextype          idx_type,
                                                 rocgraph_index_base         idx_base,
                                                 rocgraph_datatype           data_type);

rocgraph_status rocgraph_destroy_spmat_descr(rocgraph_const_spmat_descr descr);

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

rocgraph_status rocgraph_coo_aos_get(const rocgraph_spmat_descr descr,
                                     int64_t*                   rows,
                                     int64_t*                   cols,
                                     int64_t*                   nnz,
                                     void**                     coo_ind,
                                     void**                     coo_val,
                                     rocgraph_indextype*        idx_type,
                                     rocgraph_index_base*       idx_base,
                                     rocgraph_datatype*         data_type);

rocgraph_status rocgraph_const_coo_aos_get(rocgraph_const_spmat_descr descr,
                                           int64_t*                   rows,
                                           int64_t*                   cols,
                                           int64_t*                   nnz,
                                           const void**               coo_ind,
                                           const void**               coo_val,
                                           rocgraph_indextype*        idx_type,
                                           rocgraph_index_base*       idx_base,
                                           rocgraph_datatype*         data_type);

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

rocgraph_status rocgraph_ell_get(const rocgraph_spmat_descr descr,
                                 int64_t*                   rows,
                                 int64_t*                   cols,
                                 void**                     ell_col_ind,
                                 void**                     ell_val,
                                 int64_t*                   ell_width,
                                 rocgraph_indextype*        idx_type,
                                 rocgraph_index_base*       idx_base,
                                 rocgraph_datatype*         data_type);

rocgraph_status rocgraph_const_ell_get(rocgraph_const_spmat_descr descr,
                                       int64_t*                   rows,
                                       int64_t*                   cols,
                                       const void**               ell_col_ind,
                                       const void**               ell_val,
                                       int64_t*                   ell_width,
                                       rocgraph_indextype*        idx_type,
                                       rocgraph_index_base*       idx_base,
                                       rocgraph_datatype*         data_type);

rocgraph_status rocgraph_bell_get(const rocgraph_spmat_descr descr,
                                  int64_t*                   rows,
                                  int64_t*                   cols,
                                  rocgraph_direction*        ell_block_dir,
                                  int64_t*                   ell_block_dim,
                                  int64_t*                   ell_cols,
                                  void**                     ell_col_ind,
                                  void**                     ell_val,
                                  rocgraph_indextype*        idx_type,
                                  rocgraph_index_base*       idx_base,
                                  rocgraph_datatype*         data_type);

rocgraph_status rocgraph_const_bell_get(rocgraph_const_spmat_descr descr,
                                        int64_t*                   rows,
                                        int64_t*                   cols,
                                        rocgraph_direction*        ell_block_dir,
                                        int64_t*                   ell_block_dim,
                                        int64_t*                   ell_cols,
                                        const void**               ell_col_ind,
                                        const void**               ell_val,
                                        rocgraph_indextype*        idx_type,
                                        rocgraph_index_base*       idx_base,
                                        rocgraph_datatype*         data_type);

rocgraph_status rocgraph_bsr_get(const rocgraph_spmat_descr descr,
                                 int64_t*                   brows,
                                 int64_t*                   bcols,
                                 int64_t*                   bnnz,
                                 rocgraph_direction*        bdir,
                                 int64_t*                   bdim,
                                 void**                     bsr_row_ptr,
                                 void**                     bsr_col_ind,
                                 void**                     bsr_val,
                                 rocgraph_indextype*        row_ptr_type,
                                 rocgraph_indextype*        col_ind_type,
                                 rocgraph_index_base*       idx_base,
                                 rocgraph_datatype*         data_type);

rocgraph_status rocgraph_const_bsr_get(rocgraph_const_spmat_descr descr,
                                       int64_t*                   brows,
                                       int64_t*                   bcols,
                                       int64_t*                   bnnz,
                                       rocgraph_direction*        bdir,
                                       int64_t*                   bdim,
                                       const void**               bsr_row_ptr,
                                       const void**               bsr_col_ind,
                                       const void**               bsr_val,
                                       rocgraph_indextype*        row_ptr_type,
                                       rocgraph_indextype*        col_ind_type,
                                       rocgraph_index_base*       idx_base,
                                       rocgraph_datatype*         data_type);

rocgraph_status rocgraph_coo_set_pointers(rocgraph_spmat_descr descr,
                                          void*                coo_row_ind,
                                          void*                coo_col_ind,
                                          void*                coo_val);

rocgraph_status
    rocgraph_coo_aos_set_pointers(rocgraph_spmat_descr descr, void* coo_ind, void* coo_val);

rocgraph_status rocgraph_csr_set_pointers(rocgraph_spmat_descr descr,
                                          void*                csr_row_ptr,
                                          void*                csr_col_ind,
                                          void*                csr_val);

rocgraph_status rocgraph_csc_set_pointers(rocgraph_spmat_descr descr,
                                          void*                csc_col_ptr,
                                          void*                csc_row_ind,
                                          void*                csc_val);

rocgraph_status
    rocgraph_ell_set_pointers(rocgraph_spmat_descr descr, void* ell_col_ind, void* ell_val);

rocgraph_status rocgraph_bsr_set_pointers(rocgraph_spmat_descr descr,
                                          void*                bsr_row_ptr,
                                          void*                bsr_col_ind,
                                          void*                bsr_val);

rocgraph_status rocgraph_spmat_get_size(rocgraph_const_spmat_descr descr,
                                        int64_t*                   rows,
                                        int64_t*                   cols,
                                        int64_t*                   nnz);

rocgraph_status rocgraph_spmat_get_format(rocgraph_const_spmat_descr descr,
                                          rocgraph_format*           format);

rocgraph_status rocgraph_spmat_get_index_base(rocgraph_const_spmat_descr descr,
                                              rocgraph_index_base*       idx_base);

rocgraph_status rocgraph_spmat_get_values(rocgraph_spmat_descr descr, void** values);

rocgraph_status rocgraph_const_spmat_get_values(rocgraph_const_spmat_descr descr,
                                                const void**               values);

rocgraph_status rocgraph_spmat_set_values(rocgraph_spmat_descr descr, void* values);

rocgraph_status rocgraph_spmat_get_strided_batch(rocgraph_const_spmat_descr descr,
                                                 rocgraph_int*              batch_count);

rocgraph_status rocgraph_spmat_set_strided_batch(rocgraph_spmat_descr descr,
                                                 rocgraph_int         batch_count);

rocgraph_status rocgraph_coo_set_strided_batch(rocgraph_spmat_descr descr,
                                               rocgraph_int         batch_count,
                                               int64_t              batch_stride);

rocgraph_status rocgraph_csr_set_strided_batch(rocgraph_spmat_descr descr,
                                               rocgraph_int         batch_count,
                                               int64_t              offsets_batch_stride,
                                               int64_t              columns_values_batch_stride);

rocgraph_status rocgraph_csc_set_strided_batch(rocgraph_spmat_descr descr,
                                               rocgraph_int         batch_count,
                                               int64_t              offsets_batch_stride,
                                               int64_t              rows_values_batch_stride);

rocgraph_status rocgraph_spmat_get_attribute(rocgraph_const_spmat_descr descr,
                                             rocgraph_spmat_attribute   attribute,
                                             void*                      data,
                                             size_t                     data_size);

rocgraph_status rocgraph_spmat_set_attribute(rocgraph_spmat_descr     descr,
                                             rocgraph_spmat_attribute attribute,
                                             const void*              data,
                                             size_t                   data_size);

rocgraph_status rocgraph_create_dnvec_descr(rocgraph_dnvec_descr* descr,
                                            int64_t               size,
                                            void*                 values,
                                            rocgraph_datatype     data_type);

rocgraph_status rocgraph_create_const_dnvec_descr(rocgraph_const_dnvec_descr* descr,
                                                  int64_t                     size,
                                                  const void*                 values,
                                                  rocgraph_datatype           data_type);

rocgraph_status rocgraph_destroy_dnvec_descr(rocgraph_const_dnvec_descr descr);

rocgraph_status rocgraph_dnvec_get(const rocgraph_dnvec_descr descr,
                                   int64_t*                   size,
                                   void**                     values,
                                   rocgraph_datatype*         data_type);

rocgraph_status rocgraph_const_dnvec_get(rocgraph_const_dnvec_descr descr,
                                         int64_t*                   size,
                                         const void**               values,
                                         rocgraph_datatype*         data_type);

rocgraph_status rocgraph_dnvec_get_values(const rocgraph_dnvec_descr descr, void** values);

rocgraph_status rocgraph_const_dnvec_get_values(rocgraph_const_dnvec_descr descr,
                                                const void**               values);

rocgraph_status rocgraph_dnvec_set_values(rocgraph_dnvec_descr descr, void* values);

rocgraph_status rocgraph_create_dnmat_descr(rocgraph_dnmat_descr* descr,
                                            int64_t               rows,
                                            int64_t               cols,
                                            int64_t               ld,
                                            void*                 values,
                                            rocgraph_datatype     data_type,
                                            rocgraph_order        order);

rocgraph_status rocgraph_create_const_dnmat_descr(rocgraph_const_dnmat_descr* descr,
                                                  int64_t                     rows,
                                                  int64_t                     cols,
                                                  int64_t                     ld,
                                                  const void*                 values,
                                                  rocgraph_datatype           data_type,
                                                  rocgraph_order              order);

rocgraph_status rocgraph_destroy_dnmat_descr(rocgraph_const_dnmat_descr descr);

rocgraph_status rocgraph_dnmat_get(const rocgraph_dnmat_descr descr,
                                   int64_t*                   rows,
                                   int64_t*                   cols,
                                   int64_t*                   ld,
                                   void**                     values,
                                   rocgraph_datatype*         data_type,
                                   rocgraph_order*            order);

rocgraph_status rocgraph_const_dnmat_get(rocgraph_const_dnmat_descr descr,
                                         int64_t*                   rows,
                                         int64_t*                   cols,
                                         int64_t*                   ld,
                                         const void**               values,
                                         rocgraph_datatype*         data_type,
                                         rocgraph_order*            order);

rocgraph_status rocgraph_dnmat_get_values(const rocgraph_dnmat_descr descr, void** values);

rocgraph_status rocgraph_const_dnmat_get_values(rocgraph_const_dnmat_descr descr,
                                                const void**               values);

rocgraph_status rocgraph_dnmat_set_values(rocgraph_dnmat_descr descr, void* values);

rocgraph_status rocgraph_dnmat_get_strided_batch(rocgraph_const_dnmat_descr descr,
                                                 rocgraph_int*              batch_count,
                                                 int64_t*                   batch_stride);

rocgraph_status rocgraph_dnmat_set_strided_batch(rocgraph_dnmat_descr descr,
                                                 rocgraph_int         batch_count,
                                                 int64_t              batch_stride);

rocgraph_status rocgraph_coo2csr(rocgraph_handle     handle,
                                 const rocgraph_int* coo_row_ind,
                                 rocgraph_int        nnz,
                                 rocgraph_int        m,
                                 rocgraph_int*       csr_row_ptr,
                                 rocgraph_index_base idx_base);

rocgraph_status rocgraph_coosort_buffer_size(rocgraph_handle     handle,
                                             rocgraph_int        m,
                                             rocgraph_int        n,
                                             rocgraph_int        nnz,
                                             const rocgraph_int* coo_row_ind,
                                             const rocgraph_int* coo_col_ind,
                                             size_t*             buffer_size);

rocgraph_status rocgraph_coosort_by_row(rocgraph_handle handle,
                                        rocgraph_int    m,
                                        rocgraph_int    n,
                                        rocgraph_int    nnz,
                                        rocgraph_int*   coo_row_ind,
                                        rocgraph_int*   coo_col_ind,
                                        rocgraph_int*   perm,
                                        void*           temp_buffer);

rocgraph_status rocgraph_coosort_by_column(rocgraph_handle handle,
                                           rocgraph_int    m,
                                           rocgraph_int    n,
                                           rocgraph_int    nnz,
                                           rocgraph_int*   coo_row_ind,
                                           rocgraph_int*   coo_col_ind,
                                           rocgraph_int*   perm,
                                           void*           temp_buffer);

rocgraph_status rocgraph_cscsort_buffer_size(rocgraph_handle     handle,
                                             rocgraph_int        m,
                                             rocgraph_int        n,
                                             rocgraph_int        nnz,
                                             const rocgraph_int* csc_col_ptr,
                                             const rocgraph_int* csc_row_ind,
                                             size_t*             buffer_size);

rocgraph_status rocgraph_cscsort(rocgraph_handle          handle,
                                 rocgraph_int             m,
                                 rocgraph_int             n,
                                 rocgraph_int             nnz,
                                 const rocgraph_mat_descr descr,
                                 const rocgraph_int*      csc_col_ptr,
                                 rocgraph_int*            csc_row_ind,
                                 rocgraph_int*            perm,
                                 void*                    temp_buffer);

rocgraph_status rocgraph_csr2coo(rocgraph_handle     handle,
                                 const rocgraph_int* csr_row_ptr,
                                 rocgraph_int        nnz,
                                 rocgraph_int        m,
                                 rocgraph_int*       coo_row_ind,
                                 rocgraph_index_base idx_base);

rocgraph_status rocgraph_csr2csc_buffer_size(rocgraph_handle     handle,
                                             rocgraph_int        m,
                                             rocgraph_int        n,
                                             rocgraph_int        nnz,
                                             const rocgraph_int* csr_row_ptr,
                                             const rocgraph_int* csr_col_ind,
                                             rocgraph_action     copy_values,
                                             size_t*             buffer_size);

rocgraph_status rocgraph_scsr2csc(rocgraph_handle     handle,
                                  rocgraph_int        m,
                                  rocgraph_int        n,
                                  rocgraph_int        nnz,
                                  const float*        csr_val,
                                  const rocgraph_int* csr_row_ptr,
                                  const rocgraph_int* csr_col_ind,
                                  float*              csc_val,
                                  rocgraph_int*       csc_row_ind,
                                  rocgraph_int*       csc_col_ptr,
                                  rocgraph_action     copy_values,
                                  rocgraph_index_base idx_base,
                                  void*               temp_buffer);

rocgraph_status rocgraph_dcsr2csc(rocgraph_handle     handle,
                                  rocgraph_int        m,
                                  rocgraph_int        n,
                                  rocgraph_int        nnz,
                                  const double*       csr_val,
                                  const rocgraph_int* csr_row_ptr,
                                  const rocgraph_int* csr_col_ind,
                                  double*             csc_val,
                                  rocgraph_int*       csc_row_ind,
                                  rocgraph_int*       csc_col_ptr,
                                  rocgraph_action     copy_values,
                                  rocgraph_index_base idx_base,
                                  void*               temp_buffer);

rocgraph_status rocgraph_scsr2csr_compress(rocgraph_handle          handle,
                                           rocgraph_int             m,
                                           rocgraph_int             n,
                                           const rocgraph_mat_descr descr_A,
                                           const float*             csr_val_A,
                                           const rocgraph_int*      csr_row_ptr_A,
                                           const rocgraph_int*      csr_col_ind_A,
                                           rocgraph_int             nnz_A,
                                           const rocgraph_int*      nnz_per_row,
                                           float*                   csr_val_C,
                                           rocgraph_int*            csr_row_ptr_C,
                                           rocgraph_int*            csr_col_ind_C,
                                           float                    tol);

rocgraph_status rocgraph_dcsr2csr_compress(rocgraph_handle          handle,
                                           rocgraph_int             m,
                                           rocgraph_int             n,
                                           const rocgraph_mat_descr descr_A,
                                           const double*            csr_val_A,
                                           const rocgraph_int*      csr_row_ptr_A,
                                           const rocgraph_int*      csr_col_ind_A,
                                           rocgraph_int             nnz_A,
                                           const rocgraph_int*      nnz_per_row,
                                           double*                  csr_val_C,
                                           rocgraph_int*            csr_row_ptr_C,
                                           rocgraph_int*            csr_col_ind_C,
                                           double                   tol);

rocgraph_status rocgraph_csrsort_buffer_size(rocgraph_handle     handle,
                                             rocgraph_int        m,
                                             rocgraph_int        n,
                                             rocgraph_int        nnz,
                                             const rocgraph_int* csr_row_ptr,
                                             const rocgraph_int* csr_col_ind,
                                             size_t*             buffer_size);

rocgraph_status rocgraph_csrsort(rocgraph_handle          handle,
                                 rocgraph_int             m,
                                 rocgraph_int             n,
                                 rocgraph_int             nnz,
                                 const rocgraph_mat_descr descr,
                                 const rocgraph_int*      csr_row_ptr,
                                 rocgraph_int*            csr_col_ind,
                                 rocgraph_int*            perm,
                                 void*                    temp_buffer);

rocgraph_status rocgraph_sgthr(rocgraph_handle     handle,
                               rocgraph_int        nnz,
                               const float*        y,
                               float*              x_val,
                               const rocgraph_int* x_ind,
                               rocgraph_index_base idx_base);

rocgraph_status rocgraph_dgthr(rocgraph_handle     handle,
                               rocgraph_int        nnz,
                               const double*       y,
                               double*             x_val,
                               const rocgraph_int* x_ind,
                               rocgraph_index_base idx_base);

rocgraph_status
    rocgraph_create_identity_permutation(rocgraph_handle handle, rocgraph_int n, rocgraph_int* p);

rocgraph_status rocgraph_inverse_permutation(rocgraph_handle     handle,
                                             rocgraph_int        n,
                                             const rocgraph_int* p,
                                             rocgraph_int*       q,
                                             rocgraph_index_base base);

rocgraph_status rocgraph_set_identity_permutation(rocgraph_handle    handle,
                                                  int64_t            n,
                                                  void*              p,
                                                  rocgraph_indextype indextype);

rocgraph_status rocgraph_snnz_compress(rocgraph_handle          handle,
                                       rocgraph_int             m,
                                       const rocgraph_mat_descr descr_A,
                                       const float*             csr_val_A,
                                       const rocgraph_int*      csr_row_ptr_A,
                                       rocgraph_int*            nnz_per_row,
                                       rocgraph_int*            nnz_C,
                                       float                    tol);

rocgraph_status rocgraph_dnnz_compress(rocgraph_handle          handle,
                                       rocgraph_int             m,
                                       const rocgraph_mat_descr descr_A,
                                       const double*            csr_val_A,
                                       const rocgraph_int*      csr_row_ptr_A,
                                       rocgraph_int*            nnz_per_row,
                                       rocgraph_int*            nnz_C,
                                       double                   tol);

rocgraph_status rocgraph_snnz(rocgraph_handle          handle,
                              rocgraph_direction       dir,
                              rocgraph_int             m,
                              rocgraph_int             n,
                              const rocgraph_mat_descr descr,
                              const float*             A,
                              rocgraph_int             ld,
                              rocgraph_int*            nnz_per_row_columns,
                              rocgraph_int*            nnz_total_dev_host_ptr);

rocgraph_status rocgraph_dnnz(rocgraph_handle          handle,
                              rocgraph_direction       dir,
                              rocgraph_int             m,
                              rocgraph_int             n,
                              const rocgraph_mat_descr descr,
                              const double*            A,
                              rocgraph_int             ld,
                              rocgraph_int*            nnz_per_row_columns,
                              rocgraph_int*            nnz_total_dev_host_ptr);

rocgraph_status rocgraph_scoomv(rocgraph_handle          handle,
                                rocgraph_operation       trans,
                                rocgraph_int             m,
                                rocgraph_int             n,
                                rocgraph_int             nnz,
                                const float*             alpha,
                                const rocgraph_mat_descr descr,
                                const float*             coo_val,
                                const rocgraph_int*      coo_row_ind,
                                const rocgraph_int*      coo_col_ind,
                                const float*             x,
                                const float*             beta,
                                float*                   y);

rocgraph_status rocgraph_dcoomv(rocgraph_handle          handle,
                                rocgraph_operation       trans,
                                rocgraph_int             m,
                                rocgraph_int             n,
                                rocgraph_int             nnz,
                                const double*            alpha,
                                const rocgraph_mat_descr descr,
                                const double*            coo_val,
                                const rocgraph_int*      coo_row_ind,
                                const rocgraph_int*      coo_col_ind,
                                const double*            x,
                                const double*            beta,
                                double*                  y);

rocgraph_status rocgraph_scsrmv_analysis(rocgraph_handle          handle,
                                         rocgraph_operation       trans,
                                         rocgraph_int             m,
                                         rocgraph_int             n,
                                         rocgraph_int             nnz,
                                         const rocgraph_mat_descr descr,
                                         const float*             csr_val,
                                         const rocgraph_int*      csr_row_ptr,
                                         const rocgraph_int*      csr_col_ind,
                                         rocgraph_mat_info        info);

rocgraph_status rocgraph_dcsrmv_analysis(rocgraph_handle          handle,
                                         rocgraph_operation       trans,
                                         rocgraph_int             m,
                                         rocgraph_int             n,
                                         rocgraph_int             nnz,
                                         const rocgraph_mat_descr descr,
                                         const double*            csr_val,
                                         const rocgraph_int*      csr_row_ptr,
                                         const rocgraph_int*      csr_col_ind,
                                         rocgraph_mat_info        info);

rocgraph_status rocgraph_csrmv_clear(rocgraph_handle handle, rocgraph_mat_info info);

rocgraph_status rocgraph_scsrmv(rocgraph_handle          handle,
                                rocgraph_operation       trans,
                                rocgraph_int             m,
                                rocgraph_int             n,
                                rocgraph_int             nnz,
                                const float*             alpha,
                                const rocgraph_mat_descr descr,
                                const float*             csr_val,
                                const rocgraph_int*      csr_row_ptr,
                                const rocgraph_int*      csr_col_ind,
                                rocgraph_mat_info        info,
                                const float*             x,
                                const float*             beta,
                                float*                   y);

rocgraph_status rocgraph_dcsrmv(rocgraph_handle          handle,
                                rocgraph_operation       trans,
                                rocgraph_int             m,
                                rocgraph_int             n,
                                rocgraph_int             nnz,
                                const double*            alpha,
                                const rocgraph_mat_descr descr,
                                const double*            csr_val,
                                const rocgraph_int*      csr_row_ptr,
                                const rocgraph_int*      csr_col_ind,
                                rocgraph_mat_info        info,
                                const double*            x,
                                const double*            beta,
                                double*                  y);

rocgraph_status rocgraph_scsrmm(rocgraph_handle          handle,
                                rocgraph_operation       trans_A,
                                rocgraph_operation       trans_B,
                                rocgraph_int             m,
                                rocgraph_int             n,
                                rocgraph_int             k,
                                rocgraph_int             nnz,
                                const float*             alpha,
                                const rocgraph_mat_descr descr,
                                const float*             csr_val,
                                const rocgraph_int*      csr_row_ptr,
                                const rocgraph_int*      csr_col_ind,
                                const float*             B,
                                rocgraph_int             ldb,
                                const float*             beta,
                                float*                   C,
                                rocgraph_int             ldc);

rocgraph_status rocgraph_dcsrmm(rocgraph_handle          handle,
                                rocgraph_operation       trans_A,
                                rocgraph_operation       trans_B,
                                rocgraph_int             m,
                                rocgraph_int             n,
                                rocgraph_int             k,
                                rocgraph_int             nnz,
                                const double*            alpha,
                                const rocgraph_mat_descr descr,
                                const double*            csr_val,
                                const rocgraph_int*      csr_row_ptr,
                                const rocgraph_int*      csr_col_ind,
                                const double*            B,
                                rocgraph_int             ldb,
                                const double*            beta,
                                double*                  C,
                                rocgraph_int             ldc);

rocgraph_status rocgraph_scsrcolor(rocgraph_handle          handle,
                                   rocgraph_int             m,
                                   rocgraph_int             nnz,
                                   const rocgraph_mat_descr descr,
                                   const float*             csr_val,
                                   const rocgraph_int*      csr_row_ptr,
                                   const rocgraph_int*      csr_col_ind,
                                   const float*             fraction_to_color,
                                   rocgraph_int*            ncolors,
                                   rocgraph_int*            coloring,
                                   rocgraph_int*            reordering,
                                   rocgraph_mat_info        info);

rocgraph_status rocgraph_dcsrcolor(rocgraph_handle          handle,
                                   rocgraph_int             m,
                                   rocgraph_int             nnz,
                                   const rocgraph_mat_descr descr,
                                   const double*            csr_val,
                                   const rocgraph_int*      csr_row_ptr,
                                   const rocgraph_int*      csr_col_ind,
                                   const double*            fraction_to_color,
                                   rocgraph_int*            ncolors,
                                   rocgraph_int*            coloring,
                                   rocgraph_int*            reordering,
                                   rocgraph_mat_info        info);

rocgraph_status rocgraph_axpby(rocgraph_handle            handle,
                               const void*                alpha,
                               rocgraph_const_spvec_descr x,
                               const void*                beta,
                               rocgraph_dnvec_descr       y);

rocgraph_status rocgraph_dense_to_graph(rocgraph_handle             handle,
                                        rocgraph_const_dnmat_descr  mat_A,
                                        rocgraph_spmat_descr        mat_B,
                                        rocgraph_dense_to_graph_alg alg,
                                        size_t*                     buffer_size,
                                        void*                       temp_buffer);

rocgraph_status
    rocgraph_gather(rocgraph_handle handle, rocgraph_const_dnvec_descr y, rocgraph_spvec_descr x);

rocgraph_status rocgraph_graph_to_dense(rocgraph_handle             handle,
                                        rocgraph_const_spmat_descr  mat_A,
                                        rocgraph_dnmat_descr        mat_B,
                                        rocgraph_graph_to_dense_alg alg,
                                        size_t*                     buffer_size,
                                        void*                       temp_buffer);

rocgraph_status rocgraph_spmm(rocgraph_handle            handle,
                              rocgraph_operation         trans_A,
                              rocgraph_operation         trans_B,
                              const void*                alpha,
                              rocgraph_const_spmat_descr mat_A,
                              rocgraph_const_dnmat_descr mat_B,
                              const void*                beta,
                              const rocgraph_dnmat_descr mat_C,
                              rocgraph_datatype          compute_type,
                              rocgraph_spmm_alg          alg,
                              rocgraph_spmm_stage        stage,
                              size_t*                    buffer_size,
                              void*                      temp_buffer);

rocgraph_status rocgraph_spmv(rocgraph_handle            handle,
                              rocgraph_operation         trans,
                              const void*                alpha,
                              rocgraph_const_spmat_descr mat,
                              rocgraph_const_dnvec_descr x,
                              const void*                beta,
                              const rocgraph_dnvec_descr y,
                              rocgraph_datatype          compute_type,
                              rocgraph_spmv_alg          alg,
                              rocgraph_spmv_stage        stage,
                              size_t*                    buffer_size,
                              void*                      temp_buffer);

rocgraph_status rocgraph_spmv_ex(rocgraph_handle            handle,
                                 rocgraph_operation         trans,
                                 const void*                alpha,
                                 const rocgraph_spmat_descr mat,
                                 const rocgraph_dnvec_descr x,
                                 const void*                beta,
                                 const rocgraph_dnvec_descr y,
                                 rocgraph_datatype          compute_type,
                                 rocgraph_spmv_alg          alg,
                                 rocgraph_spmv_stage        stage,
                                 size_t*                    buffer_size,
                                 void*                      temp_buffer);

#ifdef __cplusplus
}
#endif
