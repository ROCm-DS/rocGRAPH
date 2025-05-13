/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph_clients_expect_eq.hpp"

#define ROCGRAPH_CLIENTS_EXPECT_TRUE(PROPOSED_) ROCGRAPH_CLIENTS_EXPECT_EQ((PROPOSED_), true)
