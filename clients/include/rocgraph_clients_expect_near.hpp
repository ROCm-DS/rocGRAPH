/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph_clients_expect_near_tolerance.hpp"

#include "rocgraph_clients_default_tolerance.hpp"

template <typename T>
inline bool rocgraph_clients_expect_near(const char* a_name,
                                         const T     a,
                                         const char* b_name,
                                         const T     b,
                                         const char* file_name,
                                         int32_t     file_line)
{
    return rocgraph_clients_expect_near_tolerance(
        a_name, a, b_name, b, rocgraph_clients_default_tolerance<T>::value, file_name, file_line);
}

#define ROCGRAPH_CLIENTS_EXPECT_NEAR(PROPOSED_, EXPECTED_)                                \
    do                                                                                    \
    {                                                                                     \
        const auto proposed_ = (PROPOSED_);                                               \
        const auto expected_ = (EXPECTED_);                                               \
        const bool eq        = rocgraph_clients_expect_near(                              \
            #(PROPOSED_), proposed_, #(EXPECTED_), expected_, __FILE__, __LINE__); \
                                                                                          \
        if(eq == false)                                                                   \
        {                                                                                 \
            ROCGRAPH_CLIENTS_FAIL();                                                      \
        }                                                                                 \
    } while(false)
