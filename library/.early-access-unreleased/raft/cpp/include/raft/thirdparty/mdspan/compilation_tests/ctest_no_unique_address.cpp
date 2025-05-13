// Copyright (2020) Sandia Corporation
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "ctest_common.hpp"

#include <experimental/mdspan>

#include <type_traits>

namespace stdex = std::experimental;

//==============================================================================
// <editor-fold desc="extents"> {{{1

MDSPAN_STATIC_TEST(sizeof(stdex::extents<size_t, 1, 2, stdex::dynamic_extent>)
                   == sizeof(ptrdiff_t));

MDSPAN_STATIC_TEST(sizeof(stdex::extents<size_t, stdex::dynamic_extent>) == sizeof(ptrdiff_t));

MDSPAN_STATIC_TEST(sizeof(stdex::extents<size_t, stdex::dynamic_extent, stdex::dynamic_extent>)
                   == 2 * sizeof(ptrdiff_t));

MDSPAN_STATIC_TEST(sizeof(stdex::extents<size_t, stdex::dynamic_extent, 1, 2, 45>)
                   == sizeof(ptrdiff_t));

MDSPAN_STATIC_TEST(sizeof(stdex::extents<size_t, 45, stdex::dynamic_extent, 1>)
                   == sizeof(ptrdiff_t));

MDSPAN_STATIC_TEST(std::is_empty<stdex::extents<size_t, 1, 2, 3>>::value);

MDSPAN_STATIC_TEST(std::is_empty<stdex::extents<size_t, 42>>::value);

// </editor-fold> end extents }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="layouts"> {{{1

MDSPAN_STATIC_TEST(
    sizeof(
        stdex::layout_left::template mapping<stdex::extents<size_t, 42, stdex::dynamic_extent, 73>>)
    == sizeof(size_t));

#if defined(__GNUC__) && (__GNUC__ > 8)
MDSPAN_STATIC_TEST(
    std::is_empty<
        stdex::layout_right::template mapping<stdex::extents<size_t, 42, 123, 73>>>::value);
#endif

MDSPAN_STATIC_TEST(sizeof(stdex::layout_stride::template mapping<
                          stdex::extents<size_t, 42, stdex::dynamic_extent, 73>>)
                   == 4 * sizeof(size_t));

// </editor-fold> end layouts }}}1
//==============================================================================
