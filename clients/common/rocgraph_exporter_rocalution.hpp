/*! \file */

// Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef ROCGRAPH_EXPORTER_ROCALUTION_HPP
#define ROCGRAPH_EXPORTER_ROCALUTION_HPP

#include "rocgraph_exporter.hpp"

class rocgraph_exporter_rocalution : public rocgraph_exporter<rocgraph_exporter_rocalution>
{
protected:
    std::string m_filename{};

public:
    ~rocgraph_exporter_rocalution();
    rocgraph_exporter_rocalution(const std::string& filename_);

    template <typename T, typename I = rocgraph_int, typename J = rocgraph_int>
    rocgraph_status write_graph_csx(rocgraph_direction dir,
                                    J                  m,
                                    J                  n,
                                    I                  nnz,
                                    const I* __restrict__ ptr,
                                    const J* __restrict__ ind,
                                    const T* __restrict__ val,
                                    rocgraph_index_base base);

    template <typename T, typename I = rocgraph_int, typename J = rocgraph_int>
    rocgraph_status write_graph_gebsx(rocgraph_direction dir,
                                      rocgraph_direction dirb,
                                      J                  mb,
                                      J                  nb,
                                      I                  nnzb,
                                      J                  block_dim_row,
                                      J                  block_dim_column,
                                      const I* __restrict__ ptr,
                                      const J* __restrict__ ind,
                                      const T* __restrict__ val,
                                      rocgraph_index_base base);

    template <typename T, typename I = rocgraph_int>
    rocgraph_status write_graph_coo(I m,
                                    I n,
                                    I nnz,
                                    const I* __restrict__ row_ind,
                                    const I* __restrict__ col_ind,
                                    const T* __restrict__ val,
                                    rocgraph_index_base base);

    template <typename T, typename I = rocgraph_int>
    rocgraph_status write_dense_vector(I size, const T* __restrict__ x, I incx);

    template <typename T, typename I = rocgraph_int>
    rocgraph_status
        write_dense_matrix(rocgraph_order order, I m, I n, const T* __restrict__ x, I ld);
};

#endif // HEADER
