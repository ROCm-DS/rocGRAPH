/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#define ROCGRAPH_CLIENTS_SKIP_TEST(A) GTEST_SKIP() << A
#else
#define ROCGRAPH_CLIENTS_SKIP_TEST(A) \
    std::cout << A << std::endl;      \
    return
#endif
