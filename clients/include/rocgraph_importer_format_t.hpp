/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef ROCGRAPH_IMPORTER_FORMAT_T_HPP
#define ROCGRAPH_IMPORTER_FORMAT_T_HPP

struct rocgraph_importer_format_t
{
#define LIST_IMPORTER_FORMATS \
    FORMAT(unknown)           \
    FORMAT(matrixmarket)      \
    FORMAT(mlcsr)             \
    FORMAT(rocalution)

    typedef enum _
    {
#define FORMAT(x_) x_,
        LIST_IMPORTER_FORMATS
    } value_type;
    static constexpr value_type all_formats[] = {LIST_IMPORTER_FORMATS};
#undef FORMAT

#define FORMAT(x_) #x_,
    static constexpr const char* s_format_names[]{LIST_IMPORTER_FORMATS};
#undef FORMAT

    value_type value{};
    rocgraph_importer_format_t() {};

public:
    static const char* extension(const value_type val)
    {
        switch(val)
        {
        case matrixmarket:
            return ".mtx";
        case mlcsr:
            return ".smtx";
        case rocalution:
            return ".csr";
        case unknown:
            return "";
        }
        return nullptr;
    }

    rocgraph_importer_format_t& operator()(const char* filename)
    {
        this->value = unknown;

        const char* ext = nullptr;
        for(const char* p = filename; *p != '\0'; ++p)
        {
            if(*p == '.')
                ext = p;
        }
        if(ext)
        {
            for(auto format : all_formats)
            {
                if(!strcmp(ext, extension(format)))
                {
                    this->value = format;
                    break;
                }
            }
        }
        return *this;
    };
};

#endif // HEADER
