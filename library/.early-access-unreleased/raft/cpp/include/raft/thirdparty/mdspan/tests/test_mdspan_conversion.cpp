// Copyright (2020) National Technology & Engineering
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include <experimental/mdspan>

#include <gtest/gtest.h>

namespace stdex = std::experimental;

TEST(TestMdspanConversionConst, test_mdspan_conversion_const)
{
    std::array<double, 6>                                 a{};
    stdex::mdspan<double, stdex::extents<uint32_t, 2, 3>> s(a.data());
    ASSERT_EQ(s.data_handle(), a.data());
    __MDSPAN_OP(s, 0, 1) = 3.14;
    stdex::mdspan<double const, stdex::extents<uint64_t, 2, 3>> c_s(s);
    ASSERT_EQ((__MDSPAN_OP(c_s, 0, 1)), 3.14);
}
