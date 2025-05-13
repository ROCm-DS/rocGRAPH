// Copyright (2019) Sandia Corporation
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "ctest_common.hpp"

#include <experimental/mdspan>

namespace stdex = std::experimental;

MDSPAN_STATIC_TEST(
    std::is_constructible<stdex::extents<size_t, 1, 2, stdex::dynamic_extent>, int>::value);

MDSPAN_STATIC_TEST(
    std::is_copy_constructible<stdex::extents<size_t, 1, 2, stdex::dynamic_extent>>::value);

MDSPAN_STATIC_TEST(std::is_copy_constructible<stdex::extents<size_t, 1, 2>>::value);

MDSPAN_STATIC_TEST(
    std::is_copy_constructible<stdex::extents<size_t, stdex::dynamic_extent>>::value);

MDSPAN_STATIC_TEST(
    std::is_move_constructible<stdex::extents<size_t, 1, 2, stdex::dynamic_extent>>::value);

MDSPAN_STATIC_TEST(std::is_default_constructible<stdex::extents<size_t, 1, 2, 3>>::value);

MDSPAN_STATIC_TEST(
    std::is_constructible<
        stdex::extents<size_t, stdex::dynamic_extent, stdex::dynamic_extent, stdex::dynamic_extent>,
        int,
        int,
        int>::value);

MDSPAN_STATIC_TEST(
    !std::is_constructible<
        stdex::extents<size_t, stdex::dynamic_extent, stdex::dynamic_extent, stdex::dynamic_extent>,
        int,
        int>::value);

MDSPAN_STATIC_TEST(
    !std::is_constructible<
        stdex::extents<size_t, stdex::dynamic_extent, stdex::dynamic_extent, stdex::dynamic_extent>,
        int>::value);

MDSPAN_STATIC_TEST(
    std::is_constructible<
        stdex::extents<size_t, stdex::dynamic_extent, stdex::dynamic_extent, stdex::dynamic_extent>,
        stdex::extents<size_t, stdex::dynamic_extent, 2, 3>>::value);

MDSPAN_STATIC_TEST(std::is_convertible<stdex::extents<size_t, 2, 3>,
                                       stdex::extents<size_t, 2, stdex::dynamic_extent>>::value);

MDSPAN_STATIC_TEST(!std::is_convertible<stdex::extents<size_t, 3, 2>,
                                        stdex::extents<size_t, 2, stdex::dynamic_extent>>::value);

MDSPAN_STATIC_TEST(std::is_constructible<stdex::extents<size_t, 2, stdex::dynamic_extent>,
                                         stdex::extents<size_t, 2, 3>>::value);

MDSPAN_STATIC_TEST(!std::is_constructible<stdex::extents<size_t, 3, stdex::dynamic_extent>,
                                          stdex::extents<size_t, 2, 3>>::value);

MDSPAN_STATIC_TEST(std::is_assignable<stdex::extents<size_t, 2, stdex::dynamic_extent>,
                                      stdex::extents<size_t, 2, 3>>::value);

MDSPAN_STATIC_TEST(std::is_same<stdex::dextents<size_t, 0>, stdex::extents<size_t>>::value);

MDSPAN_STATIC_TEST(
    std::is_same<stdex::dextents<size_t, 1>, stdex::extents<size_t, stdex::dynamic_extent>>::value);

MDSPAN_STATIC_TEST(
    std::is_same<stdex::dextents<size_t, 2>,
                 stdex::extents<size_t, stdex::dynamic_extent, stdex::dynamic_extent>>::value);

MDSPAN_STATIC_TEST(std::is_same<stdex::dextents<size_t, 3>,
                                stdex::extents<size_t,
                                               stdex::dynamic_extent,
                                               stdex::dynamic_extent,
                                               stdex::dynamic_extent>>::value);
