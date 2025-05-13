/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_status_to_string.hpp"

const char* rocgraph_status_to_string(rocgraph_status status)
{
    switch(status)
    {
    case rocgraph_status_success:
        return "rocgraph_status_success";
    case rocgraph_status_invalid_handle:
        return "rocgraph_status_invalid_handle";
    case rocgraph_status_not_implemented:
        return "rocgraph_status_not_implemented";
    case rocgraph_status_invalid_pointer:
        return "rocgraph_status_invalid_pointer";
    case rocgraph_status_invalid_size:
        return "rocgraph_status_invalid_size";
    case rocgraph_status_memory_error:
        return "rocgraph_status_memory_error";
    case rocgraph_status_internal_error:
        return "rocgraph_status_internal_error";
    case rocgraph_status_invalid_value:
        return "rocgraph_status_invalid_value";
    case rocgraph_status_arch_mismatch:
        return "rocgraph_status_arch_mismatch";
    case rocgraph_status_not_initialized:
        return "rocgraph_status_not_initialized";
    case rocgraph_status_type_mismatch:
        return "rocgraph_status_type_mismatch";
    case rocgraph_status_requires_sorted_storage:
        return "rocgraph_status_requires_sorted_storage";
    case rocgraph_status_unknown_error:
        return "rocgraph_status_unknown_error";
    case rocgraph_status_invalid_input:
        return "rocgraph_status_invalid_input";
    case rocgraph_status_unsupported_type_combination:
        return "rocgraph_status_unsupported_type_combination";
    case rocgraph_status_thrown_exception:
        return "rocgraph_status_thrown_exception";
    case rocgraph_status_continue:
        return "rocgraph_status_continue";
    }
    return "unknown";
}
