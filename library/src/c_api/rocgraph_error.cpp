// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_error.hpp"
#include "internal/aux/rocgraph_error_aux.h"

extern "C" const char* rocgraph_error_message(const rocgraph_error_t* error)
{
    if(error != nullptr)
    {
        auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_error_t const*>(error);
        return internal_pointer->error_message_.c_str();
    }
    else
    {
        return nullptr;
    }
}

extern "C" void rocgraph_error_free(rocgraph_error_t* error)
{
    if(error != nullptr)
    {
        auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_error_t const*>(error);
        delete internal_pointer;
    }
}
