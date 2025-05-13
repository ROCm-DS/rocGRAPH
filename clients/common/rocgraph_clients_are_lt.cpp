/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_clients_are_lt.hpp"

template <typename T>
bool rocgraph_clients_are_lt(const T a, const T b)
{
    return a < b;
}

#define INSTANTIATE(T) template bool rocgraph_clients_are_lt(const T a, const T b)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(int32_t);
INSTANTIATE(int64_t);
INSTANTIATE(size_t);

#undef INSTANTIATE
