/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef ROCGRAPH_TRAITS_HPP
#define ROCGRAPH_TRAITS_HPP

#include "rocgraph.h"

template <typename T>
struct floating_traits
{
    using data_t = T;
};

template <typename T>
using floating_data_t = typename floating_traits<T>::data_t;

#endif // ROCGRAPH_TRAITS_HPP
