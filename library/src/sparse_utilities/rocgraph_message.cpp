/*! \file */

// Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "control.h"
#include "debug.h"
#include "envariables.h"
#include "to_string.hpp"
#include <map>

void rocgraph::message(const char* msg_, const char* function_, const char* file_, int line_)
{
    if(rocgraph_debug_variables.get_debug_verbose())
    {
        std::cout << "// rocGRAPH.log:     { \"function\": \"" << function_ << "\"," << std::endl
                  << "//                      \"line\"    : \"" << line_ << "\"," << std::endl
                  << "//                      \"msg\"     : \"" << msg_ << "\" }" << std::endl;
    }
}

void rocgraph::warning_message(const char* msg_,
                               const char* function_,
                               const char* file_,
                               int         line_)
{
    if(rocgraph_debug_variables.get_debug_verbose())
    {
        std::cout << "// rocGRAPH.warning: { \"function\": \"" << function_ << "\"," << std::endl
                  << "//                      \"line\"    : \"" << line_ << "\"," << std::endl
                  << "//                      \"msg\"     : \"" << msg_ << "\" }" << std::endl;
    }
}

void rocgraph::error_message(
    rocgraph_status status_, const char* msg_, const char* function_, const char* file_, int line_)
{
    if(rocgraph_debug_variables.get_debug_verbose())
    {
        std::cout << "// rocGRAPH.error.trace:   { \"function\": \"" << function_ << "\","
                  << std::endl
                  << "//                            \"line\"    : \"" << line_ << "\"," << std::endl
                  << "//                            \"file\"    : \"" << file_ << "\"," << std::endl
                  << "//                            \"status\"  : \""
                  << rocgraph::to_string(status_) << "\"," << std::endl
                  << "//                            \"msg\"     : \"" << msg_ << "\" }"
                  << std::endl;
    }
}
