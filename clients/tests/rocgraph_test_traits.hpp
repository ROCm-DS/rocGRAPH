/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph_test_config.hpp"
#include "rocgraph_test_enum.hpp"

// Default traits.
template <rocgraph_test_enum::value_type ROUTINE>
struct rocgraph_test_traits : rocgraph_test_config
{
};
