/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

struct test_check
{
protected:
    static bool s_auto_testing_bad_arg;

public:
    test_check() = delete;
    static inline void set_auto_testing_bad_arg()
    {
        s_auto_testing_bad_arg = true;
    }
    static inline void reset_auto_testing_bad_arg()
    {
        s_auto_testing_bad_arg = false;
    }
    static inline bool did_auto_testing_bad_arg()
    {
        return s_auto_testing_bad_arg;
    }
};
