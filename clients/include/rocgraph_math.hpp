/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef ROCGRAPH_MATH_HPP
#define ROCGRAPH_MATH_HPP

#include "rocgraph_sparse_utility.hpp"
#include <cmath>
#include <rocgraph.h>

/* =================================================================================== */
/*! \brief  returns true if value is NaN */
template <typename T>
inline bool rocgraph_isnan(T arg);

template <>
inline bool rocgraph_isnan(int8_t arg)
{
    return false;
}

template <>
inline bool rocgraph_isnan(uint8_t arg)
{
    return false;
}

template <>
inline bool rocgraph_isnan(uint32_t arg)
{
    return false;
}

template <>
inline bool rocgraph_isnan(int arg)
{
    return false;
}

template <>
inline bool rocgraph_isnan(int64_t arg)
{
    return false;
}

template <>
inline bool rocgraph_isnan(uint64_t arg)
{
    return false;
}

template <>
inline bool rocgraph_isnan(double arg)
{
    return std::isnan(arg);
}
template <>
inline bool rocgraph_isnan(float arg)
{
    return std::isnan(arg);
}

/* =================================================================================== */
/*! \brief  returns true if value is inf */
template <typename T>
inline bool rocgraph_isinf(T arg);

template <>
inline bool rocgraph_isinf(int8_t arg)
{
    return false;
}

template <>
inline bool rocgraph_isinf(int arg)
{
    return false;
}

template <>
inline bool rocgraph_isinf(int64_t arg)
{
    return false;
}

template <>
inline bool rocgraph_isinf(uint64_t arg)
{
    return false;
}

template <>
inline bool rocgraph_isinf(float arg)
{
    return std::isinf(arg);
}

template <>
inline bool rocgraph_isinf(double arg)
{
    return std::isinf(arg);
}

/* =================================================================================== */
/*! \brief  returns complex conjugate */
template <typename T>
inline T rocgraph_conj(T arg)
{
    return arg;
}

#endif // ROCGRAPH_MATH_HPP
