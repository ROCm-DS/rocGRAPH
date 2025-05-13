/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

struct rocgraph_test_numeric_types_enum
{
    typedef enum value_type_ : int32_t
    {
        all,
        real_only,
        complex_only
    } value_type;
};
