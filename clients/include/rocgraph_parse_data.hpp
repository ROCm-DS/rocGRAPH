/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef ROCGRAPH_PARSE_DATA_HPP
#define ROCGRAPH_PARSE_DATA_HPP

#include <string>

// Parse --data and --yaml command-line arguments
bool rocgraph_parse_data(int& argc, char** argv, const std::string& default_file = "");

#endif // ROCGRAPH_PARSE_DATA_HPP
