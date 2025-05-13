/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef ROCGRAPH_LOAD_HPP
#define ROCGRAPH_LOAD_HPP

#include "rocgraph_importer_format_t.hpp"
#include "rocgraph_importer_matrixmarket.hpp"
#include "rocgraph_importer_mlbsr.hpp"
#include "rocgraph_importer_mlcsr.hpp"
#include "rocgraph_importer_rocalution.hpp"

template <rocgraph_importer_format_t::value_type IMPORTER_FORMAT>
struct rocgraph_importer_format_traits_t;

template <>
struct rocgraph_importer_format_traits_t<rocgraph_importer_format_t::rocalution>
{
    using importer_t = rocgraph_importer_rocalution;
};

template <>
struct rocgraph_importer_format_traits_t<rocgraph_importer_format_t::matrixmarket>
{
    using importer_t = rocgraph_importer_matrixmarket;
};

template <>
struct rocgraph_importer_format_traits_t<rocgraph_importer_format_t::mlbsr>
{
    using importer_t = rocgraph_importer_mlbsr;
};

template <>
struct rocgraph_importer_format_traits_t<rocgraph_importer_format_t::mlcsr>
{
    using importer_t = rocgraph_importer_mlcsr;
};

template <rocgraph_importer_format_t::value_type IMPORTER_FORMAT, typename T>
rocgraph_status rocgraph_load_template(const char* basename, const char* suffix, T& obj)
{
    using importer_t = typename rocgraph_importer_format_traits_t<IMPORTER_FORMAT>::importer_t;
    char filename[256];
    if(snprintf(filename, (size_t)256, "%s%s", basename, suffix) >= 256)
    {
        std::cerr << "rocgraph_load_template: truncated string. " << std::endl;
        return rocgraph_status_invalid_value;
    }

    importer_t importer(filename);
    importer.import(obj);
    return rocgraph_status_success;
}

template <typename T, typename... P>
rocgraph_status rocgraph_load(const char* basename, const char* suffix, T& obj, P... params);

template <rocgraph_importer_format_t::value_type IMPORTER_FORMAT, typename T, typename... P>
rocgraph_status rocgraph_load_template(const char* basename, const char* suffix, T obj, P... params)
{
    rocgraph_status status = rocgraph_load_template<IMPORTER_FORMAT>(basename, suffix, obj);
    if(status != rocgraph_status_success)
    {
        return status;
    }

    //
    // Recall dispatch.
    //
    return rocgraph_load(basename, params...);
}

//
// @brief
//
template <typename T, typename... P>
rocgraph_status rocgraph_load(const char* basename, const char* suffix, T& obj, P... params)
{
    rocgraph_importer_format_t format;
    format(suffix);
    switch(format.value)
    {
    case rocgraph_importer_format_t::unknown:
    {
        std::cerr << "unrecognized importer file format in suffix '" << suffix << "'" << std::endl;
        return rocgraph_status_invalid_value;
    }
    case rocgraph_importer_format_t::rocalution:
    {
        return rocgraph_load_template<rocgraph_importer_format_t::rocalution, T, P...>(
            basename, suffix, obj, params...);
    }
    case rocgraph_importer_format_t::matrixmarket:
    {
        return rocgraph_load_template<rocgraph_importer_format_t::matrixmarket, T, P...>(
            basename, suffix, obj, params...);
    }
    case rocgraph_importer_format_t::mlbsr:
    {
        return rocgraph_load_template<rocgraph_importer_format_t::mlbsr, T, P...>(
            basename, suffix, obj, params...);
    }
    case rocgraph_importer_format_t::mlcsr:
    {
        return rocgraph_load_template<rocgraph_importer_format_t::mlcsr, T, P...>(
            basename, suffix, obj, params...);
    }
    }
    return rocgraph_status_invalid_value;
}

#endif
