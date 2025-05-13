/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

template <typename T>
bool rocgraph_clients_expect_eq(const char* a_name,
                                const T     a,
                                const char* b_name,
                                const T     b,
                                const char* file_name,
                                int32_t     file_line);

bool rocgraph_clients_expect_eq(const char* a_name,
                                const void* a,
                                const char* b_name,
                                const void* b,
                                const char* file_name,
                                int32_t     file_line);

#define ROCGRAPH_CLIENTS_EXPECT_EQ(PROPOSED_, EXPECTED_)                              \
    do                                                                                \
    {                                                                                 \
        const auto proposed_ = (PROPOSED_);                                           \
        const auto expected_ = (EXPECTED_);                                           \
        const bool eq        = rocgraph_clients_expect_eq(                            \
            #PROPOSED_, proposed_, #EXPECTED_, expected_, __FILE__, __LINE__); \
        if(eq == false)                                                               \
        {                                                                             \
            ROCGRAPH_CLIENTS_FAIL();                                                  \
        }                                                                             \
    } while(false)
