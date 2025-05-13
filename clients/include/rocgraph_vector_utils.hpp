/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef ROCGRAPH_VECTOR_UTILS_HPP
#define ROCGRAPH_VECTOR_UTILS_HPP

#include "rocgraph_check.hpp"
#include "rocgraph_vector.hpp"

template <typename T>
struct rocgraph_vector_utils
{
    static void normalize(host_dense_vector<T>& v);
};

#endif // ROCGRAPH_VECTOR_UTILS_HPP
