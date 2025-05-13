// Copyright (C) 2020-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "structure/graph_view_impl.cuh"

namespace rocgraph
{

    // SG instantiation

    template class graph_view_t<int32_t, int32_t, true, false>;
    template class graph_view_t<int32_t, int32_t, false, false>;
    template class graph_view_t<int32_t, int64_t, true, false>;
    template class graph_view_t<int32_t, int64_t, false, false>;
    template class graph_view_t<int64_t, int64_t, true, false>;
    template class graph_view_t<int64_t, int64_t, false, false>;

} // namespace rocgraph
