// Copyright (C) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "c_api/rocgraph_array.hpp"

#include "internal/rocgraph_algorithms.h"

namespace rocgraph
{
    namespace c_api
    {

        struct rocgraph_hierarchical_clustering_result_t
        {
            double                               modularity{0};
            rocgraph_type_erased_device_array_t* vertices_{nullptr};
            rocgraph_type_erased_device_array_t* clusters_{nullptr};
        };

    } // namespace c_api
} // namespace rocgraph
