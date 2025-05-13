/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * Copyright (C) 2021-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*! \file
 */
#ifndef SPARSE_UTILITY_TYPES_H
#define SPARSE_UTILITY_TYPES_H

#include "internal/types/rocgraph_handle_t.h"
#include "internal/types/rocgraph_pointer_mode.h"
#include "internal/types/rocgraph_status.h"
#include <cmath>
#include <hip/hip_runtime.h>
#include <ostream>
#include <sstream>
#include <stddef.h>
#include <stdint.h>

/// \cond DO_NOT_DOCUMENT
#define ROCGRAPH_KERNEL_W(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_EXECUTION_UNIT) \
    __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_EXECUTION_UNIT) static __global__
#define ROCGRAPH_KERNEL(MAX_THREADS_PER_BLOCK) \
    __launch_bounds__(MAX_THREADS_PER_BLOCK) static __global__
#define ROCGRAPH_DEVICE_ILF static __device__ __forceinline__
/// \endcond

/*! \ingroup types_module
 * \typedef rocgraph_int
 *  \brief Specifies whether int32 or int64 is used.
 */
#if defined(rocgraph_ILP64)
typedef int64_t rocgraph_int;
#else
typedef int32_t rocgraph_int;
#endif

/*! \ingroup types_module
 *  \brief Descriptor of the matrix.
 *
 * \typedef rocgraph_mat_descr
 *  \details
 *  The rocGRAPH matrix descriptor is a structure holding all properties of a matrix.
 *  It must be initialized using rocgraph_create_mat_descr() and the returned
 *  descriptor must be passed to all subsequent library calls that involve the matrix.
 *  It should be destroyed at the end using rocgraph_destroy_mat_descr().
 */
typedef struct _rocgraph_mat_descr* rocgraph_mat_descr;

/*! \ingroup types_module
 *  \brief Info structure to hold all matrix meta data.
 *
 * \typedef rocgraph_mat_info
 *  \details
 *  The rocGRAPH matrix info is a structure holding all matrix information that is
 *  gathered during analysis routines. It must be initialized using
 *  rocgraph_create_mat_info() and the returned info structure must be passed to all
 *  subsequent library calls that require additional matrix information. It should be
 *  destroyed at the end using rocgraph_destroy_mat_info().
 */
typedef struct _rocgraph_mat_info* rocgraph_mat_info;

/*! \ingroup types_module
 *  \brief Generic API descriptor of the graph matrix.
 *
 * \typedef rocgraph_spmat_descr
 *  \details
 *  The rocGRAPH graph matrix descriptor is a structure holding all properties of a graph matrix.
 *  It must be initialized using rocgraph_create_coo_descr(), rocgraph_create_coo_aos_descr(),
 *  rocgraph_create_csr_descr() or  rocgraph_create_csc_descr() and the returned
 *  descriptor must be passed to all subsequent generic API library calls that involve the graph matrix.
 *  It should be destroyed at the end using rocgraph_destroy_spmat_descr().
 */
typedef struct _rocgraph_spmat_descr* rocgraph_spmat_descr;

/*! \ingroup types_module
 *  \brief Generic API descriptor of the graph matrix.
 *
 * \typedef rocgraph_const_spmat_descr
 *  \details
 *  The rocGRAPH constant graph matrix descriptor is a structure holding all properties of a graph matrix.
 *  It must be initialized using rocgraph_create__const_coo_descr(),
 *  rocgraph_create_const_csr_descr() or rocgraph_create_const_csc_descr() and the returned
 *  descriptor must be passed to all subsequent generic API library calls that involve the graph matrix.
 *  It should be destroyed at the end using rocgraph_destroy_spmat_descr().
 */
typedef struct _rocgraph_spmat_descr const* rocgraph_const_spmat_descr;

/*! \ingroup types_module
 *  \brief Generic API descriptor of the dense vector.
 *
 * \typedef rocgraph_dnvec_descr
 *  \details
 *  The rocGRAPH dense vector descriptor is a structure holding all properties of a dense vector.
 *  It must be initialized using rocgraph_create_dnvec_descr() and the returned
 *  descriptor must be passed to all subsequent generic API library calls that involve the dense vector.
 *  It should be destroyed at the end using rocgraph_destroy_dnvec_descr().
 */
typedef struct _rocgraph_dnvec_descr* rocgraph_dnvec_descr;

/*! \ingroup types_module
 *  \brief Generic API descriptor of the dense vector.
 *
 * \typedef rocgraph_const_dnvec_descr
 *  \details
 *  The rocGRAPH constant dense vector descriptor is a structure holding all properties of a dense vector.
 *  It must be initialized using rocgraph_create_const_dnvec_descr() and the returned
 *  descriptor must be passed to all subsequent generic API library calls that involve the dense vector.
 *  It should be destroyed at the end using rocgraph_destroy_dnvec_descr().
 */
typedef struct _rocgraph_dnvec_descr const* rocgraph_const_dnvec_descr;

/*! \ingroup types_module
 *  \brief Generic API descriptor of the dense matrix.
 *
 * \typedef rocgraph_dnmat_descr
 *  \details
 *  The rocGRAPH dense matrix descriptor is a structure holding all properties of a dense matrix.
 *  It must be initialized using rocgraph_create_dnmat_descr() and the returned
 *  descriptor must be passed to all subsequent generic API library calls that involve the dense matrix.
 *  It should be destroyed at the end using rocgraph_destroy_dnmat_descr().
 */
typedef struct _rocgraph_dnmat_descr* rocgraph_dnmat_descr;

/*! \ingroup types_module
 *  \brief Generic API descriptor of the dense matrix.
 *
 * \typedef rocgraph_const_dnmat_descr
 *  \details
 *  The rocGRAPH constant dense matrix descriptor is a structure holding all properties of a dense matrix.
 *  It must be initialized using rocgraph_create_const_dnmat_descr() and the returned
 *  descriptor must be passed to all subsequent generic API library calls that involve the dense matrix.
 *  It should be destroyed at the end using rocgraph_destroy_dnmat_descr().
 */
typedef struct _rocgraph_dnmat_descr const* rocgraph_const_dnmat_descr;

/*! \ingroup types_module
 *  \brief Coloring info structure to hold data gathered during analysis and later used in
 *  rocGRAPH graph matrix coloring routines.
 *
 * \typedef rocgraph_color_info
 *  \details
 *  The rocGRAPH color info is a structure holding coloring data that is
 *  gathered during analysis routines. It must be initialized using
 *  rocgraph_create_color_info() and the returned info structure must be passed to all
 *  subsequent library calls that require coloring information. It should be
 *  destroyed at the end using rocgraph_destroy_color_info().
 */
typedef struct _rocgraph_color_info* rocgraph_color_info;

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup types_module
 *  \brief Specify whether the matrix is to be transposed or not.
 *
 * \enum rocgraph_operation
 *  \details
 *  The \ref rocgraph_operation indicates the operation performed with the given matrix.
 */
typedef enum rocgraph_operation_
{
    rocgraph_operation_none                = 111, /**< Operate with matrix. */
    rocgraph_operation_transpose           = 112, /**< Operate with transpose. */
    rocgraph_operation_conjugate_transpose = 113 /**< Operate with conj. transpose. */
} rocgraph_operation;

/*! \ingroup types_module
 *  \brief Specify the matrix index base.
 *
 * \enum rocgraph_index_base
 *  \details
 *  The \ref rocgraph_index_base indicates the index base of the indices. For a
 *  given \ref rocgraph_mat_descr, the \ref rocgraph_index_base can be set using
 *  rocgraph_set_mat_index_base(). The current \ref rocgraph_index_base of a matrix
 *  can be obtained by rocgraph_get_mat_index_base().
 */
typedef enum rocgraph_index_base_
{
    rocgraph_index_base_zero = 0, /**< zero based indexing. */
    rocgraph_index_base_one  = 1 /**< one based indexing. */
} rocgraph_index_base;

/*! \ingroup types_module
 *  \brief Specify the matrix type.
 *
 * \enum rocgraph_matrix_type
 *  \details
 *  The \ref rocgraph_matrix_type indices the type of a matrix. For a given
 *  \ref rocgraph_mat_descr, the \ref rocgraph_matrix_type can be set using
 *  rocgraph_set_mat_type(). The current \ref rocgraph_matrix_type of a matrix can be
 *  obtained by rocgraph_get_mat_type().
 */
typedef enum rocgraph_matrix_type_
{
    rocgraph_matrix_type_general    = 0, /**< general matrix type. */
    rocgraph_matrix_type_symmetric  = 1, /**< symmetric matrix type. */
    rocgraph_matrix_type_hermitian  = 2, /**< hermitian matrix type. */
    rocgraph_matrix_type_triangular = 3 /**< triangular matrix type. */
} rocgraph_matrix_type;

/*! \ingroup types_module
 *  \brief Indicates if the diagonal entries are unity.
 *
 * \enum rocgraph_diag_type
 *  \details
 *  The \ref rocgraph_diag_type indicates whether the diagonal entries of a matrix are
 *  unity or not. If \ref rocgraph_diag_type_unit is specified, all present diagonal
 *  values will be ignored. For a given \ref rocgraph_mat_descr, the
 *  \ref rocgraph_diag_type can be set using rocgraph_set_mat_diag_type(). The current
 *  \ref rocgraph_diag_type of a matrix can be obtained by
 *  rocgraph_get_mat_diag_type().
 */
typedef enum rocgraph_diag_type_
{
    rocgraph_diag_type_non_unit = 0, /**< diagonal entries are non-unity. */
    rocgraph_diag_type_unit     = 1 /**< diagonal entries are unity */
} rocgraph_diag_type;

/*! \ingroup types_module
 *  \brief Specify the matrix fill mode.
 *
 * \enum rocgraph_fill_mode
 *  \details
 *  The \ref rocgraph_fill_mode indicates whether the lower or the upper part is stored
 *  in a graph triangular matrix. For a given \ref rocgraph_mat_descr, the
 *  \ref rocgraph_fill_mode can be set using rocgraph_set_mat_fill_mode(). The current
 *  \ref rocgraph_fill_mode of a matrix can be obtained by
 *  rocgraph_get_mat_fill_mode().
 */
typedef enum rocgraph_fill_mode_
{
    rocgraph_fill_mode_lower = 0, /**< lower triangular part is stored. */
    rocgraph_fill_mode_upper = 1 /**< upper triangular part is stored. */
} rocgraph_fill_mode;

/*! \ingroup types_module
 *  \brief Specify whether the matrix is stored sorted or not.
 *
 * \enum rocgraph_storage_mode
 *  \details
 *  The \ref rocgraph_storage_mode indicates whether the matrix is stored sorted or not.
 *  For a given \ref rocgraph_mat_descr, the \ref rocgraph_storage_mode can be set
 *  using rocgraph_set_mat_storage_mode(). The current \ref rocgraph_storage_mode of a
 *  matrix can be obtained by rocgraph_get_mat_storage_mode().
 */
typedef enum rocgraph_storage_mode_
{
    rocgraph_storage_mode_sorted   = 0, /**< matrix is sorted. */
    rocgraph_storage_mode_unsorted = 1 /**< matrix is unsorted. */
} rocgraph_storage_mode;

/*! \ingroup types_module
 *  \brief Specify the matrix direction.
 *
 * \enum rocgraph_direction
 *  \details
 *  The \ref rocgraph_direction indicates whether a dense matrix should be parsed by
 *  rows or by columns, assuming column-major storage.
 */
typedef enum rocgraph_direction_
{
    rocgraph_direction_row    = 0, /**< Parse the matrix by rows. */
    rocgraph_direction_column = 1 /**< Parse the matrix by columns. */
} rocgraph_direction;

/*! \ingroup types_module
 *  \brief List of rocgraph index types.
 *
 * \enum rocgraph_indextype
 *  \details
 *  Indicates the index width of a rocgraph index type.
 */
typedef enum rocgraph_indextype_
{
    rocgraph_indextype_u16 = 1, /**< 16 bit unsigned integer. */
    rocgraph_indextype_i32 = 2, /**< 32 bit signed integer. */
    rocgraph_indextype_i64 = 3 /**< 64 bit signed integer. */
} rocgraph_indextype;

/*! \ingroup types_module
 *  \brief List of rocgraph data types.
 *
 * \enum rocgraph_datatype
 *  \details
 *  Indicates the precision width of data stored in a rocgraph type.
 */
typedef enum rocgraph_datatype_
{
    rocgraph_datatype_f32_r = 151, /**< 32 bit floating point, real. */
    rocgraph_datatype_f64_r = 152, /**< 64 bit floating point, real. */
    rocgraph_datatype_i8_r  = 160, /**<  8-bit signed integer, real */
    rocgraph_datatype_u8_r  = 161, /**<  8-bit unsigned integer, real */
    rocgraph_datatype_i32_r = 162, /**< 32-bit signed integer, real */
    rocgraph_datatype_u32_r = 163 /**< 32-bit unsigned integer, real */
} rocgraph_datatype;

/*! \ingroup types_module
 *  \brief List of graph matrix formats.
 *
 * \enum rocgraph_format
 *  \details
 *  This is a list of supported \ref rocgraph_format types that are used to describe a
 *  graph matrix.
 */
typedef enum rocgraph_format_
{
    rocgraph_format_coo     = 0, /**< COO graph matrix format. */
    rocgraph_format_coo_aos = 1, /**< COO AoS graph matrix format. */
    rocgraph_format_csr     = 2, /**< CSR graph matrix format. */
    rocgraph_format_csc     = 3 /**< CSC graph matrix format. */
} rocgraph_format;

/*! \ingroup types_module
 *  \brief Specify where the operation is performed on.
 *
 * \enum rocgraph_action
 *  \details
 *  The \ref rocgraph_action indicates whether the operation is performed on the full
 *  matrix, or only on the sparsity pattern of the matrix.
 */
typedef enum rocgraph_action_
{
    rocgraph_action_symbolic = 0, /**< Operate only on indices. */
    rocgraph_action_numeric  = 1 /**< Operate on data and indices. */
} rocgraph_action;

/*! \ingroup types_module
 *  \brief List of dense matrix ordering.
 *
 * \enum rocgraph_order
 *  \details
 *  This is a list of supported \ref rocgraph_order types that are used to describe the
 *  memory layout of a dense matrix
 */
typedef enum rocgraph_order_
{
    rocgraph_order_row    = 0, /**< Row major. */
    rocgraph_order_column = 1 /**< Column major. */
} rocgraph_order;

/*! \ingroup types_module
 * \enum rocgraph_spmat_attribute
 *  \brief List of graph matrix attributes
 */
typedef enum rocgraph_spmat_attribute_
{
    rocgraph_spmat_fill_mode    = 0, /**< Fill mode attribute. */
    rocgraph_spmat_diag_type    = 1, /**< Diag type attribute. */
    rocgraph_spmat_matrix_type  = 2, /**< Matrix type attribute. */
    rocgraph_spmat_storage_mode = 3 /**< Matrix storage attribute. */
} rocgraph_spmat_attribute;

/*! \ingroup types_module
 *  \brief List of SpMV stages.
 *
 * \enum rocgraph_spmv_stage
 *  \details
 *  This is a list of possible stages during SpMV computation. Typical order is
 *  rocgraph_spmv_buffer_size, rocgraph_spmv_preprocess, rocgraph_spmv_compute.
 */
typedef enum rocgraph_spmv_stage_
{
    rocgraph_spmv_stage_buffer_size = 1, /**< Returns the required buffer size. */
    rocgraph_spmv_stage_preprocess  = 2, /**< Preprocess data. */
    rocgraph_spmv_stage_compute     = 3 /**< Performs the actual SpMV computation. */
} rocgraph_spmv_stage;

/*! \ingroup types_module
 *  \brief List of SpMV algorithms.
 *
 * \enum rocgraph_spmv_alg
 *  \details
 *  This is a list of supported \ref rocgraph_spmv_alg types that are used to perform
 *  matrix vector product.
 */
typedef enum rocgraph_spmv_alg_
{
    rocgraph_spmv_alg_default      = 0, /**< Default SpMV algorithm for the given format. */
    rocgraph_spmv_alg_coo          = 1, /**< COO SpMV algorithm 1 (segmented) for COO matrices. */
    rocgraph_spmv_alg_csr_adaptive = 2, /**< CSR SpMV algorithm 1 (adaptive) for CSR matrices. */
    rocgraph_spmv_alg_csr_stream   = 3, /**< CSR SpMV algorithm 2 (stream) for CSR matrices. */
    rocgraph_spmv_alg_coo_atomic   = 5, /**< COO SpMV algorithm 2 (atomic) for COO matrices. */
    rocgraph_spmv_alg_csr_lrb      = 7 /**< CSR SpMV algorithm 3 (LRB) for CSR matrices. */
} rocgraph_spmv_alg;

/*! \ingroup types_module
*  \brief List of SpMM algorithms.
*
 * \enum rocgraph_spmm_alg
*  \details
*  This is a list of supported \ref rocgraph_spmm_alg types that are used to perform
*  matrix vector product.
*/
typedef enum rocgraph_spmm_alg_
{
    rocgraph_spmm_alg_default = 0, /**< Default SpMM algorithm for the given format. */
    rocgraph_spmm_alg_csr, /**< SpMM algorithm for CSR format using row split and shared memory. */
    rocgraph_spmm_alg_coo_segmented, /**< SpMM algorithm for COO format using segmented scan. */
    rocgraph_spmm_alg_coo_atomic, /**< SpMM algorithm for COO format using atomics. */
    rocgraph_spmm_alg_csr_row_split, /**< SpMM algorithm for CSR format using row split and shfl. */
    rocgraph_spmm_alg_csr_merge, /**< SpMM algorithm for CSR format using nnz split algorithm. Is the same as rocgraph_spmm_alg_csr_nnz_split. */
    rocgraph_spmm_alg_coo_segmented_atomic, /**< SpMM algorithm for COO format using segmented scan and atomics. */
    rocgraph_spmm_alg_csr_merge_path, /**< SpMM algorithm for CSR format using merge path algorithm. */
    rocgraph_spmm_alg_csr_nnz_split
    = rocgraph_spmm_alg_csr_merge /**< SpMM algorithm for CSR format using nnz split algorithm. */
} rocgraph_spmm_alg;

/*! \ingroup types_module
 *  \brief List of SpMM stages.
 *
 * \enum rocgraph_spmm_stage
 *  \details
 *  This is a list of possible stages during SpMM computation. Typical order is
 *  rocgraph_spmm_buffer_size, rocgraph_spmm_preprocess, rocgraph_spmm_compute.
 */
typedef enum rocgraph_spmm_stage_
{
    rocgraph_spmm_stage_buffer_size = 1, /**< Returns the required buffer size. */
    rocgraph_spmm_stage_preprocess  = 2, /**< Preprocess data. */
    rocgraph_spmm_stage_compute     = 3 /**< Performs the actual SpMM computation. */
} rocgraph_spmm_stage;

#ifdef __cplusplus
}
#endif

#endif /* ROCGRAPH_TYPES_H */
