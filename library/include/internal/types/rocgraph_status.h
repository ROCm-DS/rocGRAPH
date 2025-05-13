// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*!\file*/
/*
 * Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 */
/*
 * Copyright (C) 2021-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
   *  @brief List of rocgraph status codes definition.
   *
   *  @details
   *  This is a list of the status types that are used by the rocGRAPH
   *  library.
   */
typedef enum
{
    rocgraph_status_success = 0,
    rocgraph_status_unknown_error,
    rocgraph_status_invalid_handle,
    rocgraph_status_invalid_input,
    rocgraph_status_invalid_pointer, /**< invalid pointer parameter. */
    rocgraph_status_invalid_size, /**< invalid size parameter. */
    rocgraph_status_not_implemented, /**< function is not implemented. */
    rocgraph_status_unsupported_type_combination, /**< type combination is not supported. */
    rocgraph_status_memory_error, /**< failed memory allocation, copy, dealloc. */
    rocgraph_status_internal_error, /**< other internal library failure. */
    rocgraph_status_invalid_value, /**< invalid value parameter. */
    rocgraph_status_arch_mismatch, /**< device arch is not supported. */
    rocgraph_status_not_initialized, /**< descriptor has not been initialized. */
    rocgraph_status_type_mismatch, /**< index types do not match. */
    rocgraph_status_requires_sorted_storage, /**< sorted storage required. */
    rocgraph_status_thrown_exception, /**< exception being thrown. */
    rocgraph_status_continue /**< Nothing preventing function to proceed */
} rocgraph_status;

#ifdef __cplusplus
}
#endif
