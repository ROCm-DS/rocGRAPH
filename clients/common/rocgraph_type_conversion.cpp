/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_type_conversion.hpp"
#include <iostream>
template <>
rocgraph_status rocgraph_type_conversion(const size_t& x, size_t& y)
{
    y = x;
    return rocgraph_status_success;
}

template <>
rocgraph_status rocgraph_type_conversion(const int32_t& x, int32_t& y)
{
    y = x;
    return rocgraph_status_success;
}

template <>
rocgraph_status rocgraph_type_conversion(const int64_t& x, int64_t& y)
{
    y = x;
    return rocgraph_status_success;
}

template <>
rocgraph_status rocgraph_type_conversion(const int32_t& x, int64_t& y)
{
    y = x;
    return rocgraph_status_success;
}

template <>
rocgraph_status rocgraph_type_conversion(const int64_t& x, size_t& y)
{
    if(x < 0)
    {
        std::cerr << "corrupted conversion from int64_t to size_t." << std::endl;
        return rocgraph_status_invalid_value;
    }
    else
    {
        y = static_cast<size_t>(x);
        return rocgraph_status_success;
    }
}

template <>
rocgraph_status rocgraph_type_conversion(const int32_t& x, size_t& y)
{
    if(x < 0)
    {
        std::cerr << "corrupted conversion from int32_t to size_t." << std::endl;
        return rocgraph_status_invalid_value;
    }
    else
    {
        y = static_cast<size_t>(x);
        return rocgraph_status_success;
    }
}

template <>
rocgraph_status rocgraph_type_conversion(const int64_t& x, int32_t& y)
{
    static constexpr int32_t int32max = std::numeric_limits<int32_t>::max();
    if(x > int32max)
    {
        std::cerr << "corrupted conversion from int64_t to int32_t." << std::endl;
        return rocgraph_status_invalid_value;
    }
    static constexpr int32_t int32min = std::numeric_limits<int32_t>::min();
    if(x < int32min)
    {
        std::cerr << "corrupted conversion from int64_t to int32_t." << std::endl;
        return rocgraph_status_invalid_value;
    }
    y = static_cast<int32_t>(x);
    return rocgraph_status_success;
}

template <>
rocgraph_status rocgraph_type_conversion(const size_t& x, int32_t& y)
{
    static constexpr int32_t int32max = std::numeric_limits<int32_t>::max();
    if(x > int32max)
    {
        std::cerr << "corrupted conversion from size_t to int32_t." << std::endl;
        return rocgraph_status_invalid_value;
    }
    y = static_cast<int32_t>(x);
    return rocgraph_status_success;
}

template <>
rocgraph_status rocgraph_type_conversion(const size_t& x, int64_t& y)
{
    static constexpr int64_t int64max = std::numeric_limits<int64_t>::max();
    if(x > int64max)
    {
        std::cerr << "corrupted conversion from size_t to int64_t." << std::endl;
        return rocgraph_status_invalid_value;
    }
    y = static_cast<int64_t>(x);
    return rocgraph_status_success;
}

template <>
rocgraph_status rocgraph_type_conversion(const float& x, double& y)
{
    y = static_cast<double>(x);
    return rocgraph_status_success;
}
