/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef TYPE_DISPATCH_HPP
#define TYPE_DISPATCH_HPP

#include "rocgraph_arguments.hpp"

// ----------------------------------------------------------------------------
// Calls TEST template based on the argument types. TEST<> is expected to
// return a functor which takes a const Arguments& argument. If the types do
// not match a recognized type combination, then TEST<void> is called.  This
// function returns the same type as TEST<...>{}(arg), usually bool or void.
// ----------------------------------------------------------------------------

// Simple functions which take only one datatype
//
// Even if the function can take mixed datatypes, this function can handle the
// cases where the types are uniform, in which case one template type argument
// is passed to TEST, and the rest are assumed to match the first.
template <template <typename...> class TEST>
auto rocgraph_simple_dispatch(const Arguments& arg)
{
    switch(arg.compute_type)
    {
    case rocgraph_datatype_f32_r:
        return TEST<float>{}(arg);
    case rocgraph_datatype_f64_r:
        return TEST<double>{}(arg);
    default:
        return TEST<void>{}(arg);
    }
}

template <template <typename...> class TEST>
auto rocgraph_it_dispatch(const Arguments& arg)
{
    const auto I = arg.index_type_I;
    if(I == rocgraph_indextype_i32)
    {
        switch(arg.compute_type)
        {
        case rocgraph_datatype_f32_r:
            return TEST<int32_t, float>{}(arg);
        case rocgraph_datatype_f64_r:
            return TEST<int32_t, double>{}(arg);
        default:
            return TEST<void>{}(arg);
        }
    }
    else if(I == rocgraph_indextype_i64)
    {
        switch(arg.compute_type)
        {
        case rocgraph_datatype_f32_r:
            return TEST<int64_t, float>{}(arg);
        case rocgraph_datatype_f64_r:
            return TEST<int64_t, double>{}(arg);
        default:
            return TEST<void>{}(arg);
        }
    }

    return TEST<void>{}(arg);
}

template <template <typename...> class TEST>
auto rocgraph_it_plus_int8_dispatch(const Arguments& arg)
{
    const auto I = arg.index_type_I;
    if(I == rocgraph_indextype_i32)
    {
        switch(arg.compute_type)
        {
        case rocgraph_datatype_i8_r:
            return TEST<int32_t, int8_t>{}(arg);
        case rocgraph_datatype_f32_r:
            return TEST<int32_t, float>{}(arg);
        case rocgraph_datatype_f64_r:
            return TEST<int32_t, double>{}(arg);
        default:
            return TEST<void>{}(arg);
        }
    }
    else if(I == rocgraph_indextype_i64)
    {
        switch(arg.compute_type)
        {
        case rocgraph_datatype_i8_r:
            return TEST<int64_t, int8_t>{}(arg);
        case rocgraph_datatype_f32_r:
            return TEST<int64_t, float>{}(arg);
        case rocgraph_datatype_f64_r:
            return TEST<int64_t, double>{}(arg);
        default:
            return TEST<void>{}(arg);
        }
    }

    return TEST<void>{}(arg);
}

template <template <typename...> class TEST>
auto rocgraph_ijt_dispatch(const Arguments& arg)
{
    const auto I = arg.index_type_I;
    const auto J = arg.index_type_J;

    if(I == rocgraph_indextype_i32 && J == rocgraph_indextype_i32)
    {
        switch(arg.compute_type)
        {
        case rocgraph_datatype_f32_r:
            return TEST<int32_t, int32_t, float>{}(arg);
        case rocgraph_datatype_f64_r:
            return TEST<int32_t, int32_t, double>{}(arg);
        default:
            return TEST<void>{}(arg);
        }
    }
    else if(I == rocgraph_indextype_i64 && J == rocgraph_indextype_i32)
    {
        switch(arg.compute_type)
        {
        case rocgraph_datatype_f32_r:
            return TEST<int64_t, int32_t, float>{}(arg);
        case rocgraph_datatype_f64_r:
            return TEST<int64_t, int32_t, double>{}(arg);
        default:
            return TEST<void>{}(arg);
        }
    }
    else if(I == rocgraph_indextype_i64 && J == rocgraph_indextype_i64)
    {
        switch(arg.compute_type)
        {
        case rocgraph_datatype_f32_r:
            return TEST<int64_t, int64_t, float>{}(arg);
        case rocgraph_datatype_f64_r:
            return TEST<int64_t, int64_t, double>{}(arg);
        default:
            return TEST<void>{}(arg);
        }
    }

    return TEST<void>{}(arg);
}

template <template <typename...> class TEST>
auto rocgraph_ixyt_dispatch(const Arguments& arg)
{
    const auto I = arg.index_type_I;

    const auto X = arg.x_type;
    const auto Y = arg.y_type;

    const auto T = arg.compute_type;

    bool f32r_case = (X == rocgraph_datatype_f32_r && X == Y && X == T);
    bool f64r_case = (X == rocgraph_datatype_f64_r && X == Y && X == T);

    bool i8r_i8r_i32r_case = (X == rocgraph_datatype_i8_r && Y == rocgraph_datatype_i8_r
                              && T == rocgraph_datatype_i32_r);

    bool i8r_i8r_f32r_case = (X == rocgraph_datatype_i8_r && Y == rocgraph_datatype_i8_r
                              && T == rocgraph_datatype_f32_r);

#define INSTANTIATE_TEST(ITYPE)                             \
    if(f32r_case)                                           \
    {                                                       \
        return TEST<ITYPE, float, float, float>{}(arg);     \
    }                                                       \
    else if(f64r_case)                                      \
    {                                                       \
        return TEST<ITYPE, double, double, double>{}(arg);  \
    }                                                       \
    else if(i8r_i8r_i32r_case)                              \
    {                                                       \
        return TEST<ITYPE, int8_t, int8_t, int32_t>{}(arg); \
    }                                                       \
    else if(i8r_i8r_f32r_case)                              \
    {                                                       \
        return TEST<ITYPE, int8_t, int8_t, float>{}(arg);   \
    }

    switch(I)
    {
    case rocgraph_indextype_u16:
    {
        return TEST<void, void, void, void>{}(arg);
    }
    case rocgraph_indextype_i32:
    {
        INSTANTIATE_TEST(int32_t);
    }
    case rocgraph_indextype_i64:
    {
        INSTANTIATE_TEST(int64_t);
    }
    }
#undef INSTANTIATE_TEST

    return TEST<void, void, void, void>{}(arg);
}

template <template <typename...> class TEST>
auto rocgraph_iaxyt_dispatch(const Arguments& arg)
{
    const auto I = arg.index_type_I;

    const auto A = arg.a_type;
    const auto X = arg.x_type;
    const auto Y = arg.y_type;

    const auto T = arg.compute_type;

    bool f32r_case = (A == rocgraph_datatype_f32_r && A == X && A == Y && A == T);
    bool f64r_case = (A == rocgraph_datatype_f64_r && A == X && A == Y && A == T);

    bool i8r_i8r_i32r_i32r_case = (A == rocgraph_datatype_i8_r && X == rocgraph_datatype_i8_r
                                   && Y == rocgraph_datatype_i32_r && T == rocgraph_datatype_i32_r);

    bool i8r_i8r_f32r_f32r_case = (A == rocgraph_datatype_i8_r && X == rocgraph_datatype_i8_r
                                   && Y == rocgraph_datatype_f32_r && T == rocgraph_datatype_f32_r);

#define INSTANTIATE_TEST(ITYPE)                                      \
    if(f32r_case)                                                    \
    {                                                                \
        return TEST<ITYPE, float, float, float, float>{}(arg);       \
    }                                                                \
    else if(f64r_case)                                               \
    {                                                                \
        return TEST<ITYPE, double, double, double, double>{}(arg);   \
    }                                                                \
    else if(i8r_i8r_i32r_i32r_case)                                  \
    {                                                                \
        return TEST<ITYPE, int8_t, int8_t, int32_t, int32_t>{}(arg); \
    }                                                                \
    else if(i8r_i8r_f32r_f32r_case)                                  \
    {                                                                \
        return TEST<ITYPE, int8_t, int8_t, float, float>{}(arg);     \
    }

    switch(I)
    {
    case rocgraph_indextype_u16:
    {
        return TEST<void, void, void, void, void>{}(arg);
    }
    case rocgraph_indextype_i32:
    {
        INSTANTIATE_TEST(int32_t);
    }
    case rocgraph_indextype_i64:
    {
        INSTANTIATE_TEST(int64_t);
    }
    }
#undef INSTANTIATE_TEST

    return TEST<void, void, void, void, void>{}(arg);
}

template <template <typename...> class TEST>
auto rocgraph_ijaxyt_dispatch(const Arguments& arg)
{
    const auto I = arg.index_type_I;
    const auto J = arg.index_type_J;

    const auto A = arg.a_type;
    const auto X = arg.x_type;
    const auto Y = arg.y_type;

    const auto T = arg.compute_type;

    bool f32r_case = (A == rocgraph_datatype_f32_r && A == X && A == Y && A == T);
    bool f64r_case = (A == rocgraph_datatype_f64_r && A == X && A == Y && A == T);

    bool i8r_i8r_i32r_i32r_case = (A == rocgraph_datatype_i8_r && X == rocgraph_datatype_i8_r
                                   && Y == rocgraph_datatype_i32_r && T == rocgraph_datatype_i32_r);

    bool i8r_i8r_f32r_f32r_case = (A == rocgraph_datatype_i8_r && X == rocgraph_datatype_i8_r
                                   && Y == rocgraph_datatype_f32_r && T == rocgraph_datatype_f32_r);

#define INSTANTIATE_TEST(ITYPE, JTYPE)                                      \
    if(f32r_case)                                                           \
    {                                                                       \
        return TEST<ITYPE, JTYPE, float, float, float, float>{}(arg);       \
    }                                                                       \
    else if(f64r_case)                                                      \
    {                                                                       \
        return TEST<ITYPE, JTYPE, double, double, double, double>{}(arg);   \
    }                                                                       \
    else if(i8r_i8r_i32r_i32r_case)                                         \
    {                                                                       \
        return TEST<ITYPE, JTYPE, int8_t, int8_t, int32_t, int32_t>{}(arg); \
    }                                                                       \
    else if(i8r_i8r_f32r_f32r_case)                                         \
    {                                                                       \
        return TEST<ITYPE, JTYPE, int8_t, int8_t, float, float>{}(arg);     \
    }

    switch(I)
    {
    case rocgraph_indextype_u16:
    {
        return TEST<void, void, void, void, void, void>{}(arg);
    }
    case rocgraph_indextype_i32:
    {
        switch(J)
        {
        case rocgraph_indextype_u16:
        case rocgraph_indextype_i64:
        {
            return TEST<void, void, void, void, void, void>{}(arg);
        }
        case rocgraph_indextype_i32:
        {
            INSTANTIATE_TEST(int32_t, int32_t);
        }
        }
    }
    case rocgraph_indextype_i64:
    {
        switch(J)
        {
        case rocgraph_indextype_u16:
        {
            return TEST<void, void, void, void, void, void>{}(arg);
        }
        case rocgraph_indextype_i32:
        {
            INSTANTIATE_TEST(int64_t, int32_t);
        }
        case rocgraph_indextype_i64:
        {
            INSTANTIATE_TEST(int64_t, int64_t);
        }
        }
    }
    }
#undef INSTANTIATE_TEST

    return TEST<void, void, void, void, void, void>{}(arg);
}

#endif // TYPE_DISPATCH_HPP
