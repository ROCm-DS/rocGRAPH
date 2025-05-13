/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph_clients_are_near_tolerance.hpp"

template <typename T>
inline bool rocgraph_clients_are_near(const T a, const T b)
{
    return rocgraph_clients_are_near_tolerance(a, b, rocgraph_clients_default_tolerance<T>::value);
}
