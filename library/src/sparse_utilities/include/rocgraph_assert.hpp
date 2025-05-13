/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "debug.h"

#ifndef NDEBUG

#define rocgraph_host_assert(cond, msg)                                                        \
    (void)((cond)                                                                              \
           || (((void)printf("%s:%s:%u: rocGRAPH failed assertion `" #cond "', message: " #msg \
                             "\n",                                                             \
                             __FILE__,                                                         \
                             __FUNCTION__,                                                     \
                             __LINE__),                                                        \
                abort()),                                                                      \
               0))

#define rocgraph_device_assert(cond, msg) rocgraph_host_assert(cond, msg)

#else

#define rocgraph_host_assert(cond, msg)                                          \
    rocgraph_debug_variables.get_debug_force_host_assert()                       \
        ? (void)((cond)                                                          \
                 || (((void)printf("%s:%s:%u: rocGRAPH failed assertion `" #cond \
                                   "', message: " #msg "\n",                     \
                                   __FILE__,                                     \
                                   __FUNCTION__,                                 \
                                   __LINE__),                                    \
                      abort()),                                                  \
                     0))                                                         \
        : (void)0

#define rocgraph_device_assert(cond, msg) ((void)0)

#endif
