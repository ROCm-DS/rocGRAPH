// Copyright (C) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "c_api/rocgraph_array.hpp"

namespace rocgraph
{
    namespace c_api
    {

        struct rocgraph_induced_subgraph_result_t
        {
            rocgraph_type_erased_device_array_t* src_{};
            rocgraph_type_erased_device_array_t* dst_{};
            rocgraph_type_erased_device_array_t* wgt_{};
            rocgraph_type_erased_device_array_t* edge_ids_{};
            rocgraph_type_erased_device_array_t* edge_type_ids_{};
            rocgraph_type_erased_device_array_t* subgraph_offsets_{};
        };

    } // namespace c_api
} // namespace rocgraph
