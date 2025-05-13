/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_clients_data_type_id_get.hpp"

template <>
rocgraph_data_type_id rocgraph_clients_data_type_id_get<float>()
{
    return rocgraph_data_type_id_float32;
}
template <>
rocgraph_data_type_id rocgraph_clients_data_type_id_get<double>()
{
    return rocgraph_data_type_id_float64;
}
template <>
rocgraph_data_type_id rocgraph_clients_data_type_id_get<int32_t>()
{
    return rocgraph_data_type_id_int32;
}
template <>
rocgraph_data_type_id rocgraph_clients_data_type_id_get<int64_t>()
{
    return rocgraph_data_type_id_int64;
}
