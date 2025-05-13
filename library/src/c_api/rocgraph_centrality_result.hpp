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

        struct rocgraph_centrality_result_t
        {
            rocgraph_type_erased_device_array_t* vertex_ids_{};
            rocgraph_type_erased_device_array_t* values_{};
            size_t                               num_iterations_{0};
            bool                                 converged_{false};
        };

        struct rocgraph_edge_centrality_result_t
        {
            rocgraph_type_erased_device_array_t* src_ids_{};
            rocgraph_type_erased_device_array_t* dst_ids_{};
            rocgraph_type_erased_device_array_t* edge_ids_{};
            rocgraph_type_erased_device_array_t* values_{};
        };

    } // namespace c_api
} // namespace rocgraph
