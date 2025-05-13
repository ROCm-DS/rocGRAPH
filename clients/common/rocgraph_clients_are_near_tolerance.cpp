/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_clients_are_near_tolerance.hpp"

template <typename T>
bool rocgraph_clients_are_near_tolerance(const T a, const T b, floating_data_t<T> tol)
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
        return std::abs(a - b) <= std::max(std::abs(a), std::abs(b)) * tol;
    }
}

#define INSTANTIATE(T) \
    template bool rocgraph_clients_are_near_tolerance(const T, const T, floating_data_t<T>)
INSTANTIATE(float);
INSTANTIATE(double);
#undef INSTANTIATE
