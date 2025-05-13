/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef ROCGRAPH_SAVE_HPP
#define ROCGRAPH_SAVE_HPP

#include "rocgraph_exporter_ascii.hpp"
#include "rocgraph_exporter_format_t.hpp"
#include "rocgraph_exporter_matrixmarket.hpp"
#include "rocgraph_exporter_rocalution.hpp"

template <rocgraph_exporter_format_t::value_type EXPORTER_FORMAT>
struct rocgraph_exporter_format_traits_t;

template <>
struct rocgraph_exporter_format_traits_t<rocgraph_exporter_format_t::rocalution>
{
    using exporter_t = rocgraph_exporter_rocalution;
};

template <>
struct rocgraph_exporter_format_traits_t<rocgraph_exporter_format_t::ascii>
{
    using exporter_t = rocgraph_exporter_ascii;
};

template <>
struct rocgraph_exporter_format_traits_t<rocgraph_exporter_format_t::matrixmarket>
{
    using exporter_t = rocgraph_exporter_matrixmarket;
};

template <rocgraph_exporter_format_t::value_type EXPORTER_FORMAT, typename T>
rocgraph_status rocgraph_save_template(const char* basename, const char* suffix, T obj)
{
    using exporter_t = typename rocgraph_exporter_format_traits_t<EXPORTER_FORMAT>::exporter_t;
    char filename[256];
    if(snprintf(filename, (size_t)256, "%s%s", basename, suffix) >= 256)
    {
        std::cerr << "rocgraph_save_template: truncated string. " << std::endl;
        return rocgraph_status_invalid_value;
    }

    exporter_t exporter(filename);
    exporter.write(obj);
    return rocgraph_status_success;
}

template <typename T, typename... P>
rocgraph_status rocgraph_save(const char* basename, const char* suffix, T obj, P... params);

template <rocgraph_exporter_format_t::value_type EXPORTER_FORMAT, typename T, typename... P>
rocgraph_status rocgraph_save_template(const char* basename, const char* suffix, T obj, P... params)
{
    rocgraph_status status = rocgraph_save_template<EXPORTER_FORMAT>(basename, suffix, obj);
    if(status != rocgraph_status_success)
    {
        return status;
    }

    //
    // Recall dispatch.
    //
    return rocgraph_save(basename, params...);
}

//
// @brief
//
template <typename T, typename... P>
rocgraph_status rocgraph_save(const char* basename, const char* suffix, T obj, P... params)
{
    rocgraph_exporter_format_t format;
    format(suffix);
    switch(format.value)
    {
    case rocgraph_exporter_format_t::unknown:
    {
        std::cerr << "unrecognized exporter file format in suffix '" << suffix << "'" << std::endl;
        return rocgraph_status_invalid_value;
    }
    case rocgraph_exporter_format_t::rocalution:
    {
        return rocgraph_save_template<rocgraph_exporter_format_t::rocalution, T, P...>(
            basename, suffix, obj, params...);
    }
    case rocgraph_exporter_format_t::matrixmarket:
    {
        return rocgraph_save_template<rocgraph_exporter_format_t::matrixmarket, T, P...>(
            basename, suffix, obj, params...);
    }
    case rocgraph_exporter_format_t::ascii:
    {
        return rocgraph_save_template<rocgraph_exporter_format_t::ascii, T, P...>(
            basename, suffix, obj, params...);
    }
    }
    return rocgraph_status_invalid_value;
}

#endif
