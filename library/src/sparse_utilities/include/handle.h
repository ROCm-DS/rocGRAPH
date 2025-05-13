/*! \file */

/*
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "c_api/rocgraph_handle.hpp"
#include "sparse_utility_auxiliary.h"
#include <fstream>
#include <hip/hip_runtime_api.h>

typedef struct _rocgraph_csrmv_info* rocgraph_csrmv_info;

/********************************************************************************
 * \brief rocgraph_mat_descr is a structure holding the rocgraph matrix
 * descriptor. It must be initialized using rocgraph_create_mat_descr()
 * and the retured handle must be passed to all subsequent library function
 * calls that involve the matrix.
 * It should be destroyed at the end using rocgraph_destroy_mat_descr().
 *******************************************************************************/
struct _rocgraph_mat_descr
{
    // matrix type
    rocgraph_matrix_type type = rocgraph_matrix_type_general;
    // fill mode
    rocgraph_fill_mode fill_mode = rocgraph_fill_mode_lower;
    // diagonal type
    rocgraph_diag_type diag_type = rocgraph_diag_type_non_unit;
    // index base
    rocgraph_index_base base = rocgraph_index_base_zero;
    // storage mode
    rocgraph_storage_mode storage_mode = rocgraph_storage_mode_sorted;
    // maximum nnz per row
    int64_t max_nnz_per_row = 0;
};

/********************************************************************************
 * \brief rocgraph_mat_info is a structure holding the matrix info data that is
 * gathered during the analysis routines. It must be initialized by calling
 * rocgraph_create_mat_info() and the returned info structure must be passed
 * to all subsequent function calls that require additional information. It
 * should be destroyed at the end using rocgraph_destroy_mat_info().
 *******************************************************************************/
struct _rocgraph_mat_info
{
    rocgraph_csrmv_info csrmv_info{};
};

/********************************************************************************
 * \brief rocgraph_color_info is a structure holding the color info data that is
 * gathered during the analysis routines. It must be initialized by calling
 * rocgraph_create_color_info() and the returned info structure must be passed
 * to all subsequent function calls that require additional information. It
 * should be destroyed at the end using rocgraph_destroy_color_info().
 *******************************************************************************/
struct _rocgraph_color_info
{
};

struct rocgraph_adaptive_info
{
    size_t    size{}; // num row blocks
    void*     row_blocks{};
    uint32_t* wg_flags{};
    void*     wg_ids{};
};

struct rocgraph_lrb_info
{
    void* rows_offsets_scratch{}; // size of m
    void* rows_bins{}; // size of m
    void* n_rows_bins{}; // size of 32

    size_t    size{};
    uint32_t* wg_flags{};

    int64_t nRowsBins[32]{}; // host array
};

/********************************************************************************
 * \brief rocgraph_csrmv_info is a structure holding the rocgraph csrmv info
 * data gathered during csrmv_analysis. It must be initialized using the
 * create_csrmv_info() routine. It should be destroyed at the end
 * destroy_csrmv_info().
 *******************************************************************************/
struct _rocgraph_csrmv_info
{
    rocgraph_adaptive_info adaptive;
    rocgraph_lrb_info      lrb;

    // some data to verify correct execution
    rocgraph_operation         trans = rocgraph_operation_none;
    int64_t                    m{};
    int64_t                    n{};
    int64_t                    nnz{};
    int64_t                    max_rows{};
    const _rocgraph_mat_descr* descr{};
    const void*                csr_row_ptr{};
    const void*                csr_col_ind{};

    rocgraph_indextype index_type_I = rocgraph_indextype_u16;
    rocgraph_indextype index_type_J = rocgraph_indextype_u16;
};

namespace rocgraph
{
    /********************************************************************************
 * \brief rocgraph_csrmv_info is a structure holding the rocgraph csrmv info
 * data gathered during csrmv_analysis. It must be initialized using the
 * create_csrmv_info() routine. It should be destroyed at the end
 * using destroy_csrmv_info().
 *******************************************************************************/
    rocgraph_status create_csrmv_info(rocgraph_csrmv_info* info);

    /********************************************************************************
 * \brief Copy csrmv info.
 *******************************************************************************/
    rocgraph_status copy_csrmv_info(rocgraph_csrmv_info dest, const rocgraph_csrmv_info src);

    /********************************************************************************
 * \brief Destroy csrmv info.
 *******************************************************************************/
    rocgraph_status destroy_csrmv_info(rocgraph_csrmv_info info);
}

struct _rocgraph_spmat_descr
{
    bool init{};

    mutable bool analysed{};

    int64_t rows{};
    int64_t cols{};
    int64_t nnz{};

    void* row_data{};
    void* col_data{};
    void* ind_data{};
    void* val_data{};

    const void* const_row_data{};
    const void* const_col_data{};
    const void* const_ind_data{};
    const void* const_val_data{};

    rocgraph_indextype row_type{};
    rocgraph_indextype col_type{};
    rocgraph_datatype  data_type{};

    rocgraph_index_base idx_base{};
    rocgraph_format     format{};

    rocgraph_mat_descr descr{};
    rocgraph_mat_info  info{};
};

struct _rocgraph_dnvec_descr
{
    bool init{};

    int64_t           size{};
    void*             values{};
    const void*       const_values{};
    rocgraph_datatype data_type{};
};

struct _rocgraph_dnmat_descr
{
    bool init{};

    int64_t rows{};
    int64_t cols{};
    int64_t ld{};

    void* values{};

    const void* const_values{};

    rocgraph_datatype data_type{};
    rocgraph_order    order{};

    int64_t batch_count{};
    int64_t batch_stride{};
};

namespace rocgraph
{
    //
    // Get architecture name.
    //
    std::string handle_get_arch_name(rocgraph_handle handle);

    struct rocpsarse_arch_names
    {
        static constexpr const char* gfx908 = "gfx908";
    };

    //
    // Get xnack mode.
    //
    std::string handle_get_xnack_mode(rocgraph_handle handle);
}
