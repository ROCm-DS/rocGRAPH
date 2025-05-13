/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph_test_template.hpp"

template <rocgraph_test_enum::value_type ROUTINE, rocgraph_test_dispatch_enum::value_type DISPATCH>
struct rocgraph_test_template_traits;

template <rocgraph_test_enum::value_type ROUTINE>
struct rocgraph_test_template_traits<ROUTINE, rocgraph_test_dispatch_enum::t>
{
    using filter = typename rocgraph_test_t_template<ROUTINE>::test;
    template <typename... P>
    using caller = typename rocgraph_test_t_template<ROUTINE>::template test_call<P...>;
};

template <rocgraph_test_enum::value_type ROUTINE>
struct rocgraph_test_template_traits<ROUTINE, rocgraph_test_dispatch_enum::it>
{
    using filter = typename rocgraph_test_it_template<ROUTINE>::test;
    template <typename... P>
    using caller = typename rocgraph_test_it_template<ROUTINE>::template test_call<P...>;
};

template <rocgraph_test_enum::value_type ROUTINE>
struct rocgraph_test_template_traits<ROUTINE, rocgraph_test_dispatch_enum::it_plus_int8>
{
    using filter = typename rocgraph_test_it_plus_int8_template<ROUTINE>::test;
    template <typename... P>
    using caller = typename rocgraph_test_it_plus_int8_template<ROUTINE>::template test_call<P...>;
};

template <rocgraph_test_enum::value_type ROUTINE>
struct rocgraph_test_template_traits<ROUTINE, rocgraph_test_dispatch_enum::ijt>
{
    using filter = typename rocgraph_test_ijt_template<ROUTINE>::test;
    template <typename... P>
    using caller = typename rocgraph_test_ijt_template<ROUTINE>::template test_call<P...>;
};

template <rocgraph_test_enum::value_type ROUTINE>
struct rocgraph_test_template_traits<ROUTINE, rocgraph_test_dispatch_enum::ixyt>
{
    using filter = typename rocgraph_test_ixyt_template<ROUTINE>::test;
    template <typename... P>
    using caller = typename rocgraph_test_ixyt_template<ROUTINE>::template test_call<P...>;
};

template <rocgraph_test_enum::value_type ROUTINE>
struct rocgraph_test_template_traits<ROUTINE, rocgraph_test_dispatch_enum::iaxyt>
{
    using filter = typename rocgraph_test_iaxyt_template<ROUTINE>::test;
    template <typename... P>
    using caller = typename rocgraph_test_iaxyt_template<ROUTINE>::template test_call<P...>;
};

template <rocgraph_test_enum::value_type ROUTINE>
struct rocgraph_test_template_traits<ROUTINE, rocgraph_test_dispatch_enum::ijaxyt>
{
    using filter = typename rocgraph_test_ijaxyt_template<ROUTINE>::test;
    template <typename... P>
    using caller = typename rocgraph_test_ijaxyt_template<ROUTINE>::template test_call<P...>;
};
