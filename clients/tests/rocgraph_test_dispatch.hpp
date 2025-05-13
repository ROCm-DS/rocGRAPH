/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph_test.hpp"
#include "rocgraph_test_dispatch_enum.hpp"
#include "type_dispatch.hpp"

template <rocgraph_test_dispatch_enum::value_type TYPE_DISPATCH = rocgraph_test_dispatch_enum::t>
struct rocgraph_test_dispatch;

template <>
struct rocgraph_test_dispatch<rocgraph_test_dispatch_enum::t>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocgraph_simple_dispatch<TEST>(arg);
    }
};

template <>
struct rocgraph_test_dispatch<rocgraph_test_dispatch_enum::it>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocgraph_it_dispatch<TEST>(arg);
    }
};

template <>
struct rocgraph_test_dispatch<rocgraph_test_dispatch_enum::it_plus_int8>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocgraph_it_plus_int8_dispatch<TEST>(arg);
    }
};

template <>
struct rocgraph_test_dispatch<rocgraph_test_dispatch_enum::ijt>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocgraph_ijt_dispatch<TEST>(arg);
    }
};

template <>
struct rocgraph_test_dispatch<rocgraph_test_dispatch_enum::ixyt>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocgraph_ixyt_dispatch<TEST>(arg);
    }
};

template <>
struct rocgraph_test_dispatch<rocgraph_test_dispatch_enum::iaxyt>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocgraph_iaxyt_dispatch<TEST>(arg);
    }
};

template <>
struct rocgraph_test_dispatch<rocgraph_test_dispatch_enum::ijaxyt>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocgraph_ijaxyt_dispatch<TEST>(arg);
    }
};
