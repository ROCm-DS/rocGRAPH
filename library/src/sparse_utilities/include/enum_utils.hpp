/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

namespace rocgraph
{
    //
    // Provide some utility methods for enums.
    //
    struct enum_utils
    {
        template <typename U>
        static bool is_invalid(U value_);
        template <typename U>
        static const char* to_string(U value_);
    };
}
