/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph_importer.hpp"

template <typename I, typename J, typename T, typename IMPORTER>
rocgraph_status rocgraph_import_graph_csr(rocgraph_importer<IMPORTER>& importer,
                                          host_dense_vector<I>&        row_ptr,
                                          host_dense_vector<J>&        col_ind,
                                          host_dense_vector<T>&        val,
                                          J&                           M,
                                          J&                           N,
                                          I&                           nnz,
                                          rocgraph_index_base          base)
{
    rocgraph_direction  dir;
    rocgraph_index_base import_base;
    rocgraph_status     status = importer.import_graph_csx(&dir, &M, &N, &nnz, &import_base);
    if(status != rocgraph_status_success)
    {
        return status;
    }
    if(dir != rocgraph_direction_row)
    {
        std::cerr << "expected csr matrix " << std::endl;
        return rocgraph_status_invalid_value;
    }

    row_ptr.resize(M + 1);
    col_ind.resize(nnz);
    val.resize(nnz);

    status = importer.import_graph_csx(row_ptr.data(), col_ind.data(), val.data());
    if(status != rocgraph_status_success)
    {
        return status;
    }

    status = rocgraph_importer_switch_base(M + 1, row_ptr, import_base, base);
    if(status != rocgraph_status_success)
    {
        return status;
    }

    status = rocgraph_importer_switch_base(nnz, col_ind, import_base, base);
    if(status != rocgraph_status_success)
    {
        return status;
    }

    return rocgraph_status_success;
}

template <typename I, typename T, typename IMPORTER>
rocgraph_status rocgraph_import_graph_coo(rocgraph_importer<IMPORTER>& importer,
                                          host_dense_vector<I>&        row_ind,
                                          host_dense_vector<I>&        col_ind,
                                          host_dense_vector<T>&        val,
                                          I&                           M,
                                          I&                           N,
                                          int64_t&                     nnz,
                                          rocgraph_index_base          base)
{

    rocgraph_index_base import_base;
    rocgraph_status     status = importer.import_graph_coo(&M, &N, &nnz, &import_base);
    if(status != rocgraph_status_success)
    {
        return status;
    }

    row_ind.resize(nnz);
    col_ind.resize(nnz);
    val.resize(nnz);

    status = importer.import_graph_coo(row_ind.data(), col_ind.data(), val.data());
    if(status != rocgraph_status_success)
    {
        return status;
    }

    status = rocgraph_importer_switch_base(nnz, row_ind, import_base, base);
    if(status != rocgraph_status_success)
    {
        return status;
    }
    status = rocgraph_importer_switch_base(nnz, col_ind, import_base, base);
    if(status != rocgraph_status_success)
    {
        return status;
    }

    return rocgraph_status_success;
}
