// Copyright (2020) National Technology & Engineering
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include <experimental/mdspan>

#include <gtest/gtest.h>

namespace stdex = std::experimental;

TEST(TestElementAccess, element_access_with_std_array)
{
    std::array<double, 6>                               a{};
    stdex::mdspan<double, stdex::extents<size_t, 2, 3>> s(a.data());
    ASSERT_EQ(__MDSPAN_OP(s, (std::array<int, 2>{1, 2})), 0);
    __MDSPAN_OP(s, (std::array<int, 2>{0, 1})) = 3.14;
    ASSERT_EQ(__MDSPAN_OP(s, (std::array<int, 2>{0, 1})), 3.14);
}
