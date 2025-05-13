// Copyright (2019) Sandia Corporation
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "macros.hpp"

#include <cstddef> // size_t
#include <limits> // numeric_limits

namespace std
{
    namespace experimental
    {

        _MDSPAN_INLINE_VARIABLE constexpr auto dynamic_extent = std::numeric_limits<size_t>::max();

        namespace detail
        {

            template <class>
            constexpr auto __make_dynamic_extent()
            {
                return dynamic_extent;
            }

            template <size_t>
            constexpr auto __make_dynamic_extent_integral()
            {
                return dynamic_extent;
            }

        } // end namespace detail

    } // end namespace experimental
} // namespace std

//==============================================================================================================
