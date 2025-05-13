/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph-types.h"
#include "rocgraph_clients_expect_array_eq.hpp"
#include "rocgraph_clients_expect_eq.hpp"
#include "rocgraph_clients_expect_false.hpp"
#include "rocgraph_clients_expect_gt.hpp"
#include "rocgraph_clients_expect_indirect_array_eq.hpp"
#include "rocgraph_clients_expect_le.hpp"
#include "rocgraph_clients_expect_lt.hpp"
#include "rocgraph_clients_expect_ne.hpp"
#include "rocgraph_clients_expect_true.hpp"
#include "rocgraph_sparse_utility.hpp"
#include <cstdint>

template <typename T>
void rocgraph_clients_unit_check_general(
    size_t m, size_t n, const T* a, size_t lda, const T* b, size_t ldb);

template <typename T>
void rocgraph_clients_unit_check_enum(const T a, const T b);

template <typename T, typename I>
void rocgraph_clients_unit_check_array_indirect(
    size_t M, const T* a, size_t a_inc, const I* a_perm, const T* b, size_t b_inc, const I* b_perm);

template <typename T>
void rocgraph_clients_expect_array_lt_scalar(size_t size, const T* a, T s);

template <typename T>
inline void rocgraph_clients_unit_check_scalar(const T a, const T b)
{
    rocgraph_clients_unit_check_general(1, 1, &a, 1, &b, 1);
}

template <typename T>
inline void rocgraph_clients_unit_check_segments(size_t n, const T* a, const T* b)
{
    rocgraph_clients_unit_check_general(1, n, a, 1, b, 1);
}
