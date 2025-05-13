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

        struct rocgraph_labeling_result_t
        {
            rocgraph_type_erased_device_array_t* vertex_ids_;
            rocgraph_type_erased_device_array_t* labels_;
        };

    } // namespace c_api
} // namespace rocgraph
