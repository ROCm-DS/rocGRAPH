// Copyright (2020) Sandia Corporation
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "ctest_common.hpp"

#include <experimental/mdspan>

#include <type_traits>

namespace stdex = std::experimental;

//==============================================================================
// <editor-fold desc="mdspan"> {{{1

MDSPAN_STATIC_TEST(
    std::is_convertible<stdex::mdspan<double, stdex::dextents<size_t, 1>>,
                        stdex::mdspan<double const, stdex::dextents<size_t, 1>>>::value);

MDSPAN_STATIC_TEST(!std::is_convertible<stdex::mdspan<double const, stdex::dextents<size_t, 1>>,
                                        stdex::mdspan<double, stdex::dextents<size_t, 1>>>::value);

// </editor-fold> end mdspan }}}1
//==============================================================================
