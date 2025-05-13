/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

template <typename T>
struct rocgraph_clients_default_tolerance;

template <>
struct rocgraph_clients_default_tolerance<int32_t>
{
    static constexpr int32_t value = 0;
};

template <>
struct rocgraph_clients_default_tolerance<float>
{
    static constexpr float value = 1.0e-3f;
};

template <>
struct rocgraph_clients_default_tolerance<double>
{
    static constexpr double value = 1.0e-10;
};
