/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

struct rocgraph_test_dispatch_enum
{
    typedef enum value_type_ : int32_t
    {
        t,
        it,
        it_plus_int8,
        ijt,
        ixyt,
        iaxyt,
        ijaxyt
    } value_type;
};
