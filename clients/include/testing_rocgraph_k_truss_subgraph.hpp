/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph_arguments.hpp"

template <typename T>
void testing_rocgraph_k_truss_subgraph_bad_arg(const Arguments& arg);
void testing_rocgraph_k_truss_subgraph_extra(const Arguments& arg);
template <typename T>
void testing_rocgraph_k_truss_subgraph(const Arguments& arg);
