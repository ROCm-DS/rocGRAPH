// Copyright (2019) Sandia Corporation
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "macros.hpp"

namespace std
{
    namespace experimental
    {

        struct full_extent_t
        {
            explicit full_extent_t() = default;
        };

        _MDSPAN_INLINE_VARIABLE constexpr auto full_extent = full_extent_t{};

    } // end namespace experimental
} // namespace std
