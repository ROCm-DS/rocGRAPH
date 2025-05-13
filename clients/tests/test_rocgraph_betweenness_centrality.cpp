/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "test.hpp"
#include "testing_rocgraph_betweenness_centrality.hpp"

TEST_ROUTINE_WITH_CONFIG(rocgraph_betweenness_centrality, c_api, rocgraph_test_config_it, arg.M);
