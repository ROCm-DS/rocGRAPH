// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "c_api/rocgraph_array.hpp"

namespace rocgraph
{
    namespace c_api
    {

        struct rocgraph_vertex_pairs_t
        {
            rocgraph_type_erased_device_array_t* first_;
            rocgraph_type_erased_device_array_t* second_;
        };

    } // namespace c_api
} // namespace rocgraph
