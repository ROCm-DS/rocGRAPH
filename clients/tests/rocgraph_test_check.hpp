/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph_test_traits.hpp"

template <rocgraph_test_enum::value_type ROUTINE>
struct rocgraph_test_check
{
private:
    template <typename T>
    static inline constexpr bool is_valid_type()
    {
        switch(rocgraph_test_traits<ROUTINE>::s_numeric_types)
        {
        case rocgraph_test_numeric_types_enum::all:
        {
            return std::is_same<T, int8_t>{} || std::is_same<T, float>{}
                   || std::is_same<T, double>{};
        }
        case rocgraph_test_numeric_types_enum::real_only:
        {
            return std::is_same<T, int8_t>{} || std::is_same<T, float>{}
                   || std::is_same<T, double>{};
        }
        case rocgraph_test_numeric_types_enum::complex_only:
        {
            return false;
        }
        }
        return false;
    };

    template <typename T, typename... P>
    static inline constexpr bool is_valid_type_list_check()
    {
        constexpr std::size_t n = sizeof...(P);
        if(n == 0)
        {
            //
            // last type name.
            //
            return is_valid_type<T>();
        }
        else
        {
            if(!std::is_same<T, int32_t>{} && !std::is_same<T, int64_t>{})
            {
                return false;
            }
            return is_valid_type_list<P...>();
        }
    }

    template <typename... Targs>
    static inline constexpr bool is_valid_type_list()
    {
        return is_valid_type_list_check<Targs...>();
    }

    template <>
    static inline constexpr bool is_valid_type_list<>()
    {
        return false;
    }

public:
    template <typename... P>
    static constexpr bool is_type_valid()
    {
        return is_valid_type_list<P...>();
    }
};
