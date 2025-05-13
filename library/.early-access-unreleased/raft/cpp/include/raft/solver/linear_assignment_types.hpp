// Copyright (c) 2020-2022, NVIDIA CORPORATION.
// Copyright 2020 KETAN DATE & RAKESH NAGI
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

namespace raft::solver
{
    template <typename vertex_t, typename weight_t>
    struct Vertices
    {
        vertex_t* row_assignments;
        vertex_t* col_assignments;
        int*      row_covers;
        int*      col_covers;
        weight_t* row_duals;
        weight_t* col_duals;
        weight_t* col_slacks;
    };

    template <typename vertex_t>
    struct VertexData
    {
        vertex_t* parents;
        vertex_t* children;
        int*      is_visited;
    };
} // namespace raft::solver
