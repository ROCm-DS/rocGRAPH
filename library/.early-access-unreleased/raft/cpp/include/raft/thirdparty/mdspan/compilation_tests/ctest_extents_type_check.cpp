// Copyright (2019) Sandia Corporation
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "ctest_common.hpp"

#include <experimental/mdspan>

namespace stdex = std::experimental;

using E1 = stdex::extents<int32_t, stdex::dynamic_extent, 3>;

MDSPAN_STATIC_TEST(
    std::is_same<typename E1::index_type, int32_t>::value&&
                        std::is_same<typename E1::size_type, uint32_t>::value&&
                        std::is_same<typename E1::rank_type, size_t>::value&&
                        std::is_same<decltype(E1::rank()), typename E1::rank_type>::value&&
                        std::is_same<decltype(E1::rank_dynamic()), typename E1::rank_type>::value&&
                        std::is_same<decltype(E1::static_extent(0)), size_t>::value&&
                                    std::is_same<decltype(E1::static_extent(1)), size_t>::value&&
                                    std::is_same<decltype(std::declval<E1>().extent(0)),
                                                 typename E1::index_type>::value&&
                                    std::is_same<decltype(std::declval<E1>().extent(1)),
                                                 typename E1::index_type>::value
    && (E1::rank() == 2) && (E1::rank_dynamic() == 1)
    && (E1::static_extent(0) == stdex::dynamic_extent) && (E1::static_extent(1) == 3));

using E2 = stdex::extents<int64_t, stdex::dynamic_extent, 3, stdex::dynamic_extent>;

MDSPAN_STATIC_TEST(
    std::is_same<typename E2::index_type, int64_t>::value&&
                        std::is_same<typename E2::size_type, uint64_t>::value&&
                        std::is_same<typename E2::rank_type, size_t>::value&&
                        std::is_same<decltype(E2::rank()), typename E2::rank_type>::value&&
                        std::is_same<decltype(E2::rank_dynamic()), typename E2::rank_type>::value&&
                        std::is_same<decltype(E2::static_extent(0)), size_t>::value&&
                                    std::is_same<decltype(E2::static_extent(1)), size_t>::value&&
                                    std::is_same<decltype(E2::static_extent(2)), size_t>::value&&
                                    std::is_same<decltype(std::declval<E2>().extent(0)),
                                                 typename E2::index_type>::value&&
                                            std::is_same<decltype(std::declval<E2>().extent(1)),
                                                         typename E2::index_type>::value&&
                                            std::is_same<decltype(std::declval<E2>().extent(2)),
                                                         typename E2::index_type>::value
    && (E2::rank() == 3) && (E2::rank_dynamic() == 2)
    && (E2::static_extent(0) == stdex::dynamic_extent) && (E2::static_extent(1) == 3)
    && (E2::static_extent(2) == stdex::dynamic_extent));
