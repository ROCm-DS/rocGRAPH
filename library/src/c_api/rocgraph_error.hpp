// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "internal/types/rocgraph_error_t.h"
#include "internal/types/rocgraph_status.h"

#include <string>

#define CAPI_EXPECTS(STATEMENT, ERROR_CODE, ERROR_MESSAGE, ERROR_OBJECT) \
    {                                                                    \
        if(!(STATEMENT))                                                 \
        {                                                                \
            (ERROR_OBJECT) = reinterpret_cast<rocgraph_error_t*>(        \
                new rocgraph::c_api::rocgraph_error_t{ERROR_MESSAGE});   \
            return (ERROR_CODE);                                         \
        }                                                                \
    }

namespace rocgraph
{
    namespace c_api
    {

        struct rocgraph_error_t
        {
            std::string error_message_{};

            rocgraph_error_t(const char* what)
                : error_message_(what)
            {
            }
        };

    } // namespace c_api
} // namespace rocgraph
