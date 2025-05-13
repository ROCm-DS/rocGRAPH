/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph_clients_fail.hpp"
#include "rocgraph_traits.hpp"

template <typename T, typename I>
bool rocgraph_clients_expect_indirect_array_near_tolerance(size_t      size,
                                                           const char* a_name,
                                                           const T* __restrict__ a,
                                                           size_t      a_inc,
                                                           const char* a_perm_name,
                                                           const I* __restrict__ a_perm,
                                                           const char* b_name,
                                                           const T* __restrict__ b,
                                                           size_t      b_inc,
                                                           const char* b_perm_name,
                                                           const I* __restrict__ b_perm,
                                                           floating_data_t<T> tolerance,
                                                           const char*        file_name,
                                                           int32_t            file_line);

#define ROCGRAPH_CLIENTS_EXPECT_INDIRECT_ARRAY_NEAR_TOLERANCE(SIZE_,                                       \
                                                              PROPOSED_,                                   \
                                                              PROPOSED_INC_,                               \
                                                              PROPOSED_PERM_,                              \
                                                              EXPECTED_,                                   \
                                                              EXPECTED_INC_,                               \
                                                              EXPECTED_PERM_,                              \
                                                              TOLERANCE_)                                  \
    do                                                                                                     \
    {                                                                                                      \
        const auto size_          = (SIZE_);                                                               \
        const auto proposed_      = (PROPOSED_);                                                           \
        const auto proposed_inc_  = (PROPOSED_INC_);                                                       \
        const auto proposed_perm_ = (PROPOSED_PERM_);                                                      \
        const auto expected_      = (EXPECTED_);                                                           \
        const auto expected_inc_  = (EXPECTED_INC_);                                                       \
        const auto expected_perm_ = (EXPECTED_PERM_);                                                      \
        const auto tolerance_     = (TOLERANCE_);                                                          \
        const bool eq             = rocgraph_clients_expect_indirect_array_near_tolerance(size_,           \
                                                                              #PROPOSED_,      \
                                                                              proposed_,       \
                                                                              proposed_inc_,   \
                                                                              #PROPOSED_PERM_, \
                                                                              proposed_perm_,  \
                                                                              #EXPECTED_,      \
                                                                              expected_,       \
                                                                              expected_inc_,   \
                                                                              #EXPECTED_PERM_, \
                                                                              expected_perm_,  \
                                                                              tolerance_,      \
                                                                              __FILE__,        \
                                                                              __LINE__);       \
        if(eq == false)                                                                                    \
        {                                                                                                  \
            ROCGRAPH_CLIENTS_FAIL();                                                                       \
        }                                                                                                  \
    } while(false)

#define ROCGRAPH_CLIENTS_EXPECT_INDIRECT_SEGMENT_NEAR_TOLERANCE(                                           \
    SIZE_, PROPOSED_, PROPOSED_PERM_, EXPECTED_, EXPECTED_PERM_, TOLERANCE_)                               \
    do                                                                                                     \
    {                                                                                                      \
        const auto size_          = (SIZE_);                                                               \
        const auto proposed_      = (PROPOSED_);                                                           \
        const auto expected_      = (EXPECTED_);                                                           \
        const auto proposed_perm_ = (PROPOSED_PERM_);                                                      \
        const auto expected_perm_ = (EXPECTED_PERM_);                                                      \
        const auto tolerance_     = (TOLERANCE_);                                                          \
        const bool eq             = rocgraph_clients_expect_indirect_array_near_tolerance(size_,           \
                                                                              #PROPOSED_,      \
                                                                              proposed_,       \
                                                                              1,               \
                                                                              #PROPOSED_PERM_, \
                                                                              proposed_perm_,  \
                                                                              #EXPECTED_,      \
                                                                              expected_,       \
                                                                              1,               \
                                                                              #EXPECTED_PERM_, \
                                                                              expected_perm_,  \
                                                                              tolerance_,      \
                                                                              __FILE__,        \
                                                                              __LINE__);       \
        if(eq == false)                                                                                    \
        {                                                                                                  \
            ROCGRAPH_CLIENTS_FAIL();                                                                       \
        }                                                                                                  \
    } while(false)
