/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph_traits.hpp"

template <typename T>
bool rocgraph_clients_are_near_tolerance(const T a, const T b, floating_data_t<T> tol);
