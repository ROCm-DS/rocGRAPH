// Copyright (2020) Sandia Corporation
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "ctest_common.hpp"

#include <experimental/mdspan>

#include <type_traits>

namespace stdex = std::experimental;

//==============================================================================
// <editor-fold desc="helper utilities"> {{{1

MDSPAN_STATIC_TEST(
    !std::is_base_of<stdex::extents<int, 1, 2, 3>,
                     stdex::detail::__partially_static_sizes<int, size_t, 1, 2, 3>>::value);

MDSPAN_STATIC_TEST(!std::is_base_of<stdex::detail::__partially_static_sizes<int, size_t, 1, 2, 3>,
                                    stdex::extents<int, 1, 2, 3>>::value);

MDSPAN_STATIC_TEST(std::is_trivially_copyable<
                   stdex::detail::__partially_static_sizes<int, size_t, 1, 2, 3>>::value);

// </editor-fold> end helper utilities }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="extents"> {{{1

MDSPAN_STATIC_TEST(
    std::is_trivially_copyable<stdex::extents<size_t, 1, 2, stdex::dynamic_extent>>::value);

MDSPAN_STATIC_TEST(
    std::is_trivially_copyable<stdex::extents<size_t, stdex::dynamic_extent>>::value);

MDSPAN_STATIC_TEST(std::is_trivially_copyable<
                   stdex::extents<size_t, stdex::dynamic_extent, stdex::dynamic_extent>>::value);

MDSPAN_STATIC_TEST(
    std::is_trivially_copyable<stdex::extents<size_t, stdex::dynamic_extent, 1, 2, 45>>::value);

MDSPAN_STATIC_TEST(
    std::is_trivially_copyable<stdex::extents<size_t, 45, stdex::dynamic_extent, 1>>::value);

MDSPAN_STATIC_TEST(std::is_trivially_copyable<stdex::extents<size_t, 1, 2, 3>>::value);

MDSPAN_STATIC_TEST(std::is_trivially_copyable<stdex::extents<size_t, 42>>::value);

// </editor-fold> end extents }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="layouts"> {{{1

MDSPAN_STATIC_TEST(std::is_trivially_copyable<stdex::layout_left::template mapping<
                       stdex::extents<size_t, 42, stdex::dynamic_extent, 73>>>::value);

MDSPAN_STATIC_TEST(std::is_trivially_copyable<stdex::layout_right::template mapping<
                       stdex::extents<size_t, 42, stdex::dynamic_extent, 73>>>::value);

MDSPAN_STATIC_TEST(
    std::is_trivially_copyable<stdex::layout_right::template mapping<
        stdex::extents<size_t, stdex::dynamic_extent, stdex::dynamic_extent>>>::value);

MDSPAN_STATIC_TEST(std::is_trivially_copyable<stdex::layout_stride::template mapping<
                       stdex::extents<size_t, 42, stdex::dynamic_extent, 73>>>::value);

MDSPAN_STATIC_TEST(
    std::is_trivially_copyable<
        stdex::layout_stride::template mapping<stdex::extents<size_t, 42, 27, 73>>>::value);

MDSPAN_STATIC_TEST(
    std::is_trivially_copyable<stdex::layout_stride::template mapping<
        stdex::extents<size_t, stdex::dynamic_extent, stdex::dynamic_extent>>>::value);

struct layout_stride_as_member_should_be_standard_layout
    : stdex::layout_stride::template mapping<stdex::extents<size_t, 1, 2, 3>>
{
    int foo;
};

// Fails with MSVC which adds some padding
#ifndef _MDSPAN_COMPILER_MSVC
MDSPAN_STATIC_TEST(
    std::is_trivially_copyable<layout_stride_as_member_should_be_standard_layout>::value);
#endif

// </editor-fold> end layouts }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="mdspan"> {{{1

MDSPAN_STATIC_TEST(
    std::is_trivially_copyable<stdex::mdspan<double, stdex::extents<size_t, 1, 2, 3>>>::value);

MDSPAN_STATIC_TEST(
    std::is_trivially_copyable<stdex::mdspan<int, stdex::dextents<size_t, 2>>>::value);

MDSPAN_STATIC_TEST(
    std::is_trivially_copyable<
        stdex::mdspan<double,
                      stdex::extents<size_t, stdex::dynamic_extent, stdex::dynamic_extent>,
                      stdex::layout_left,
                      stdex::default_accessor<double>>>::value);

// </editor-fold> end mdspan }}}1
//==============================================================================
