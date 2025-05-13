/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph-types.h"

template <typename X, typename Y>
rocgraph_status rocgraph_type_conversion(const X& x, Y& y);
