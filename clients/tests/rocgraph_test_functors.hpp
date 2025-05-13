/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph_test_enum.hpp"
#include "rocgraph_test_utility.hpp"

template <rocgraph_test_enum::value_type ROUTINE>
struct rocgraph_test_functors
{
    static std::string name_suffix(const Arguments& arg);
};
