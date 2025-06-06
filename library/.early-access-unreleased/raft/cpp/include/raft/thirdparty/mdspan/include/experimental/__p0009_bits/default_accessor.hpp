// Copyright (2019) Sandia Corporation
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "macros.hpp"

#include <cstddef> // size_t

namespace std
{
    namespace experimental
    {

        template <class ElementType>
        struct default_accessor
        {

            using offset_policy    = default_accessor;
            using element_type     = ElementType;
            using reference        = ElementType&;
            using data_handle_type = ElementType*;

            MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr default_accessor() noexcept = default;

            MDSPAN_TEMPLATE_REQUIRES(class OtherElementType,
                                     /* requires */ (_MDSPAN_TRAIT(is_convertible,
                                                                   OtherElementType (*)[],
                                                                   element_type (*)[])))
            MDSPAN_INLINE_FUNCTION
            constexpr default_accessor(default_accessor<OtherElementType>) noexcept {}

            MDSPAN_INLINE_FUNCTION
            constexpr data_handle_type offset(data_handle_type p, size_t i) const noexcept
            {
                return p + i;
            }

            MDSPAN_FORCE_INLINE_FUNCTION
            constexpr reference access(data_handle_type p, size_t i) const noexcept
            {
                return p[i];
            }
        };

    } // end namespace experimental
} // end namespace std
