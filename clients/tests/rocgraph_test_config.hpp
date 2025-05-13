/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph_test_dispatch_enum.hpp"
#include "rocgraph_test_numeric_types_enum.hpp"

template <rocgraph_test_dispatch_enum::value_type      DISPATCH,
          rocgraph_test_numeric_types_enum::value_type NUMERIC_TYPES>
struct rocgraph_test_config_template
{
    static constexpr rocgraph_test_dispatch_enum::value_type      s_dispatch      = DISPATCH;
    static constexpr rocgraph_test_numeric_types_enum::value_type s_numeric_types = NUMERIC_TYPES;
};

struct rocgraph_test_config : rocgraph_test_config_template<rocgraph_test_dispatch_enum::t,
                                                            rocgraph_test_numeric_types_enum::all>
{
};

struct rocgraph_test_config_real_only
    : rocgraph_test_config_template<rocgraph_test_dispatch_enum::t,
                                    rocgraph_test_numeric_types_enum::real_only>
{
};

struct rocgraph_test_config_complex_only
    : rocgraph_test_config_template<rocgraph_test_dispatch_enum::t,
                                    rocgraph_test_numeric_types_enum::complex_only>
{
};

struct rocgraph_test_config_it
    : rocgraph_test_config_template<rocgraph_test_dispatch_enum::it,
                                    rocgraph_test_numeric_types_enum::all>
{
};

struct rocgraph_test_config_it_real_only
    : rocgraph_test_config_template<rocgraph_test_dispatch_enum::it,
                                    rocgraph_test_numeric_types_enum::real_only>
{
};

struct rocgraph_test_config_it_complex_only
    : rocgraph_test_config_template<rocgraph_test_dispatch_enum::it,
                                    rocgraph_test_numeric_types_enum::complex_only>
{
};

struct rocgraph_test_config_it_plus_int8
    : rocgraph_test_config_template<rocgraph_test_dispatch_enum::it_plus_int8,
                                    rocgraph_test_numeric_types_enum::all>
{
};

struct rocgraph_test_config_ijt
    : rocgraph_test_config_template<rocgraph_test_dispatch_enum::ijt,
                                    rocgraph_test_numeric_types_enum::all>
{
};

struct rocgraph_test_config_ijt_real_only
    : rocgraph_test_config_template<rocgraph_test_dispatch_enum::ijt,
                                    rocgraph_test_numeric_types_enum::real_only>
{
};

struct rocgraph_test_config_ijt_complex_only
    : rocgraph_test_config_template<rocgraph_test_dispatch_enum::ijt,
                                    rocgraph_test_numeric_types_enum::complex_only>
{
};

struct rocgraph_test_config_ixyt
    : rocgraph_test_config_template<rocgraph_test_dispatch_enum::ixyt,
                                    rocgraph_test_numeric_types_enum::all>
{
};

struct rocgraph_test_config_iaxyt
    : rocgraph_test_config_template<rocgraph_test_dispatch_enum::iaxyt,
                                    rocgraph_test_numeric_types_enum::all>
{
};

struct rocgraph_test_config_ijaxyt
    : rocgraph_test_config_template<rocgraph_test_dispatch_enum::ijaxyt,
                                    rocgraph_test_numeric_types_enum::all>
{
};
