/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#include "rocgraph_test_enum.hpp"

namespace
{
    template <rocgraph_test_enum::value_type ROUTINE>
    struct rocgraph_test_call
    {
        template <typename...>
        static void testing_bad_arg(const Arguments& arg);
        template <typename...>
        static void testing(const Arguments& arg);
        static void testing_extra(const Arguments& arg);
    };
}
