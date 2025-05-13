/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph_clients_fail.hpp"

template <typename T>
bool rocgraph_clients_expect_gt(const char* a_name,
                                const T     a,
                                const char* b_name,
                                const T     b,
                                const char* file_name,
                                int32_t     file_line);

#define ROCGRAPH_CLIENTS_EXPECT_GT(PROPOSED_, EXPECTED_)                              \
    do                                                                                \
    {                                                                                 \
        const auto proposed_ = (PROPOSED_);                                           \
        const auto expected_ = (EXPECTED_);                                           \
        const bool eq        = rocgraph_clients_expect_gt(                            \
            #PROPOSED_, proposed_, #EXPECTED_, expected_, __FILE__, __LINE__); \
        if(eq == false)                                                               \
        {                                                                             \
            ROCGRAPH_CLIENTS_FAIL();                                                  \
        }                                                                             \
    } while(false)
