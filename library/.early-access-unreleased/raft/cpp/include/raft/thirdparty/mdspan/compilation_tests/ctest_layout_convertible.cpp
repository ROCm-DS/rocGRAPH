// Copyright (2019) Sandia Corporation
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "ctest_common.hpp"

#include <experimental/mdspan>

namespace stdex = std::experimental;

struct NotARealLayout
{
    template <class Extents>
    struct mapping
    {
        using extents_type = Extents;
        using rank_type    = typename extents_type::rank_type;
        using index_type   = typename extents_type::index_type;
        using layout_type  = NotARealLayout;

        constexpr extents_type& extents() const
        {
            return ext;
        }

        template <class... Idx>
        index_type operator()(Idx...) const
        {
            return 0;
        }

        index_type required_span_size() const
        {
            return 0;
        }

        index_type stride(rank_type) const
        {
            return 1;
        }

    private:
        extents_type ext;
    };
};

template <bool unique>
struct AStridedLayout
{
    template <class Extents>
    struct mapping
    {
        using extents_type = Extents;
        using rank_type    = typename extents_type::rank_type;
        using index_type   = typename extents_type::index_type;
        using layout_type  = AStridedLayout;

        constexpr extents_type& extents() const
        {
            return ext;
        }

        template <class... Idx>
        index_type operator()(Idx...) const
        {
            return 0;
        }

        index_type required_span_size() const
        {
            return 0;
        }

        index_type stride(rank_type) const
        {
            return 1;
        }

        constexpr static bool is_always_strided()
        {
            return true;
        }
        constexpr static bool is_always_unique()
        {
            return unique;
        }
        constexpr static bool is_always_exhaustive()
        {
            return true;
        }
        constexpr bool is_strided()
        {
            return true;
        }
        constexpr bool is_unique()
        {
            return unique;
        }
        constexpr bool is_exhaustive()
        {
            return true;
        }

    private:
        extents_type ext;
    };
};

using E1  = stdex::extents<int32_t, 2, 2>;
using E2  = stdex::extents<int64_t, 2, 2>;
using LS1 = stdex::layout_stride::mapping<E1>;
using LS2 = stdex::layout_stride::mapping<E2>;

MDSPAN_STATIC_TEST(!std::is_constructible<LS1, AStridedLayout<false>::mapping<E2>>::value
                   && !std::is_convertible<AStridedLayout<false>::mapping<E2>, LS1>::value);

MDSPAN_STATIC_TEST(std::is_constructible<LS2, AStridedLayout<true>::mapping<E1>>::value&&
                       std::is_convertible<AStridedLayout<true>::mapping<E1>, LS2>::value);

MDSPAN_STATIC_TEST(!std::is_constructible<LS1, NotARealLayout::mapping<E2>>::value);
