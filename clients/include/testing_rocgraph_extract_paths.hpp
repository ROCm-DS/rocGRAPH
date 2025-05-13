/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph_arguments.hpp"

template <typename T>
void testing_rocgraph_extract_paths_bad_arg(const Arguments& arg);
void testing_rocgraph_extract_paths_extra(const Arguments& arg);
template <typename T>
void testing_rocgraph_extract_paths(const Arguments& arg);
