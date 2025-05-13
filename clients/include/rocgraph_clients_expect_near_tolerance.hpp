/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph_traits.hpp"

template <typename T>
bool rocgraph_clients_expect_near_tolerance(const char*        a_name,
                                            const T            a,
                                            const char*        b_name,
                                            const T            b,
                                            floating_data_t<T> tol,
                                            const char*        file_name,
                                            int32_t            file_line);

#define ROCGRAPH_CLIENTS_EXPECT_NEAR_TOLERANCE(PROPOSED_, EXPECTED_, TOLERANCE_)                   \
    do                                                                                             \
    {                                                                                              \
        const auto proposed_  = (PROPOSED_);                                                       \
        const auto expected_  = (EXPECTED_);                                                       \
        const auto tolerance_ = (TOLERANCE_);                                                      \
        const bool eq         = rocgraph_clients_expect_near_tolerance(                            \
            #PROPOSED_, proposed_, #EXPECTED_, expected_, tolerance_, __FILE__, __LINE__); \
        if(eq == false)                                                                            \
        {                                                                                          \
            ROCGRAPH_CLIENTS_FAIL();                                                               \
        }                                                                                          \
    } while(false)
