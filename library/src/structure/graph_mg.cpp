// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "structure/graph_impl.cuh"

namespace rocgraph
{

    // MG instantiation

    template class graph_t<int32_t, int32_t, true, true>;
    template class graph_t<int32_t, int32_t, false, true>;
    template class graph_t<int32_t, int64_t, true, true>;
    template class graph_t<int32_t, int64_t, false, true>;
    template class graph_t<int64_t, int64_t, true, true>;
    template class graph_t<int64_t, int64_t, false, true>;

} // namespace rocgraph
