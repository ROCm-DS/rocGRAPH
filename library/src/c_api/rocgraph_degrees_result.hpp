// Copyright (C) 2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "c_api/rocgraph_array.hpp"

namespace rocgraph
{
    namespace c_api
    {

        struct rocgraph_degrees_result_t
        {
            bool                                 is_symmetric{false};
            rocgraph_type_erased_device_array_t* vertex_ids_{};
            rocgraph_type_erased_device_array_t* in_degrees_{};
            rocgraph_type_erased_device_array_t* out_degrees_{};
        };

    } // namespace c_api
} // namespace rocgraph
