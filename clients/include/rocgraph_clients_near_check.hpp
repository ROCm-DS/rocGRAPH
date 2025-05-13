/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph_clients_are_near_tolerance.hpp"
#include "rocgraph_clients_default_tolerance.hpp"
#include "rocgraph_clients_expect_array_near_tolerance.hpp"
#include "rocgraph_clients_expect_indirect_array_near_tolerance.hpp"
#include "rocgraph_clients_expect_near_tolerance.hpp"
#include "rocgraph_traits.hpp"

template <typename T>
void rocgraph_clients_near_check_general(size_t             m,
                                         size_t             n,
                                         const T*           a,
                                         size_t             lda,
                                         const T*           b,
                                         size_t             ldb,
                                         floating_data_t<T> tol
                                         = rocgraph_clients_default_tolerance<T>::value);

template <typename T, typename I>
void rocgraph_clients_near_check_array_indirect(size_t             M,
                                                const T*           a,
                                                size_t             a_inc,
                                                const I*           a_perm,
                                                const T*           b,
                                                size_t             b_inc,
                                                const I*           b_perm,
                                                floating_data_t<T> tol
                                                = rocgraph_clients_default_tolerance<T>::value);

template <typename T>
void rocgraph_clients_near_check_array(size_t             M,
                                       const T*           a,
                                       size_t             a_inc,
                                       const T*           b,
                                       size_t             b_inc,
                                       floating_data_t<T> tol
                                       = rocgraph_clients_default_tolerance<T>::value);

template <typename T>
inline void rocgraph_clients_near_check_scalar(const T*           a,
                                               const T*           b,
                                               floating_data_t<T> tol
                                               = rocgraph_clients_default_tolerance<T>::value)
{
    rocgraph_clients_near_check_general(1, 1, a, 1, b, 1, tol);
}

template <typename T>
inline void rocgraph_clients_near_check_segments(size_t             n,
                                                 const T*           a,
                                                 const T*           b,
                                                 floating_data_t<T> tol
                                                 = rocgraph_clients_default_tolerance<T>::value)
{
    rocgraph_clients_near_check_general(1, n, a, 1, b, 1, tol);
}
