/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph_clients_fail.hpp"

template <typename T>
bool rocgraph_clients_expect_array_eq(size_t      size,
                                      const char* a_name,
                                      const T* __restrict__ a,
                                      size_t      a_inc,
                                      const char* b_name,
                                      const T* __restrict__ b,
                                      size_t      b_inc,
                                      const char* file_name,
                                      int32_t     file_line);

#define ROCGRAPH_CLIENTS_EXPECT_ARRAY_EQ(                                          \
    SIZE_, PROPOSED_, PROPOSED_INC_, EXPECTED_, EXPECTED_INC_)                     \
    do                                                                             \
    {                                                                              \
        const auto size_         = (SIZE_);                                        \
        const auto proposed_     = (PROPOSED_);                                    \
        const auto proposed_inc_ = (PROPOSED_INC_);                                \
        const auto expected_     = (EXPECTED_);                                    \
        const auto expected_inc_ = (EXPECTED_INC_);                                \
        const bool eq            = rocgraph_clients_expect_array_eq(size_,         \
                                                         #PROPOSED_,    \
                                                         proposed_,     \
                                                         proposed_inc_, \
                                                         #EXPECTED_,    \
                                                         expected_,     \
                                                         expected_inc_, \
                                                         __FILE__,      \
                                                         __LINE__);     \
        if(eq == false)                                                            \
        {                                                                          \
            ROCGRAPH_CLIENTS_FAIL();                                               \
        }                                                                          \
    } while(false)

#define ROCGRAPH_CLIENTS_EXPECT_SEGMENT_EQ(SIZE_, PROPOSED_, EXPECTED_)                            \
    do                                                                                             \
    {                                                                                              \
        const auto size_     = (SIZE_);                                                            \
        const auto proposed_ = (PROPOSED_);                                                        \
        const auto expected_ = (EXPECTED_);                                                        \
        const bool eq        = rocgraph_clients_expect_array_eq(                                   \
            size_, #PROPOSED_, proposed_, 1, #EXPECTED_, expected_, 1, __FILE__, __LINE__); \
        if(eq == false)                                                                            \
        {                                                                                          \
            ROCGRAPH_CLIENTS_FAIL();                                                               \
        }                                                                                          \
    } while(false)
