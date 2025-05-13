/*! \file */

/*
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "internal/types/rocgraph_status.h"

namespace rocgraph
{
    //
    // Log a message.
    //
    void message(const char* msg_, const char* function_, const char* file_, int line_);

    //
    // Log a warning message.
    //
    void warning_message(const char* msg_, const char* function_, const char* file_, int line_);

    //
    // Log an error message.
    //
    void error_message(rocgraph_status status_,
                       const char*     msg_,
                       const char*     function_,
                       const char*     file_,
                       int             line_);

#define ROCGRAPH_MESSAGE(MESSAGE__) rocgraph::message(MESSAGE__, __FUNCTION__, __FILE__, __LINE__)
#define ROCGRAPH_WARNING_MESSAGE(MESSAGE__) \
    rocgraph::warning_message(MESSAGE__, __FUNCTION__, __FILE__, __LINE__)
#define ROCGRAPH_ERROR_MESSAGE(STATUS__, MESSAGE__) \
    rocgraph::error_message(STATUS__, MESSAGE__, __FUNCTION__, __FILE__, __LINE__)
}
