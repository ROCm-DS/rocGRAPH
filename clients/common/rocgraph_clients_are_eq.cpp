/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_clients_are_eq.hpp"

template <typename T>
bool rocgraph_clients_are_eq(const T a, const T b)
{
    return a == b;
}

template <>
bool rocgraph_clients_are_eq(const float a, const float b)
{
    if(std::isnan(a))
    {
        return std::isnan(b);
    }
    else if(std::isinf(a))
    {
        return std::isinf(b);
    }
    else
    {
        return a == b;
    }
}

template <>
bool rocgraph_clients_are_eq(const double a, const double b)
{
    if(std::isnan(a))
    {
        return std::isnan(b);
    }
    else if(std::isinf(a))
    {
        return std::isinf(b);
    }
    else
    {
        return a == b;
    }
}

#define INSTANTIATE(T) template bool rocgraph_clients_are_eq(const T a, const T b)
INSTANTIATE(int32_t);
INSTANTIATE(int64_t);
INSTANTIATE(size_t);
INSTANTIATE(bool);

#include "rocgraph-types.h"
INSTANTIATE(rocgraph_status);

#undef INSTANTIATE
