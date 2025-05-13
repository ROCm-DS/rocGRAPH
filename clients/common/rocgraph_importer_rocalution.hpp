/*! \file */

// Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef ROCGRAPH_IMPORTER_ROCALUTION_HPP
#define ROCGRAPH_IMPORTER_ROCALUTION_HPP

#include "rocgraph_importer.hpp"

class rocgraph_importer_rocalution : public rocgraph_importer<rocgraph_importer_rocalution>
{
protected:
    std::string m_filename;

public:
    using IMPL = rocgraph_importer_rocalution;
    rocgraph_importer_rocalution(const std::string& filename_);

public:
    template <typename I = rocgraph_int, typename J = rocgraph_int>
    rocgraph_status import_graph_gebsx(rocgraph_direction*  dir,
                                       rocgraph_direction*  dirb,
                                       J*                   mb,
                                       J*                   nb,
                                       I*                   nnzb,
                                       J*                   block_dim_row,
                                       J*                   block_dim_column,
                                       rocgraph_index_base* base);
    template <typename T, typename I = rocgraph_int, typename J = rocgraph_int>
    rocgraph_status import_graph_gebsx(I* ptr, J* ind, T* val);
    template <typename I = rocgraph_int>
    rocgraph_status import_graph_coo(I* m, I* n, int64_t* nnz, rocgraph_index_base* base);
    template <typename T, typename I = rocgraph_int>
    rocgraph_status import_graph_coo(I* row_ind, I* col_ind, T* val);
    template <typename I = rocgraph_int, typename J = rocgraph_int>
    rocgraph_status
        import_graph_csx(rocgraph_direction* dir, J* m, J* n, I* nnz, rocgraph_index_base* base);

    template <typename T, typename I = rocgraph_int, typename J = rocgraph_int>
    rocgraph_status import_graph_csx(I* ptr, J* ind, T* val);

private:
    struct info_csx
    {
        size_t         m{};
        size_t         nnz{};
        std::ifstream* in{};
    };
    info_csx m_info_csx{};

public:
};

#endif
