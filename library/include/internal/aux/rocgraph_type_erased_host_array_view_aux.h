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

#include <stddef.h>

#include "internal/types/rocgraph_byte_t.h"
#include "internal/types/rocgraph_type_erased_host_array_t.h"
#include "internal/types/rocgraph_type_erased_host_array_view_t.h"
#include "rocgraph-export.h"

/** @ingroup aux_module
 */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief    Create a type erased host array view from
 *           a raw host pointer.
 *
 * @param [in]  pointer     Raw host pointer
 * @param [in]  n_elems     The number of elements in the array
 * @param [in]  dtype       The type of array to create
 * @return pointer to the view of the host array
 */
ROCGRAPH_EXPORT rocgraph_type_erased_host_array_view_t* rocgraph_type_erased_host_array_view_create(
    void* pointer, size_t n_elems, rocgraph_data_type_id dtype);

/**
 * @brief    Destroy a type erased host array view
 *
 * @param [in]  p    Pointer to the type erased host array view
 */
ROCGRAPH_EXPORT void
    rocgraph_type_erased_host_array_view_free(rocgraph_type_erased_host_array_view_t* p);

/**
 * @brief    Get the size of a type erased host array view
 *
 * @param [in]  p    Pointer to the type erased host array view
 * @return The number of elements in the array
 */
ROCGRAPH_EXPORT size_t
    rocgraph_type_erased_host_array_size(const rocgraph_type_erased_host_array_view_t* p);

/**
 * @brief    Get the type of a type erased host array view
 *
 * @param [in]  p    Pointer to the type erased host array view
 * @return The type of the elements in the array
 */
ROCGRAPH_EXPORT rocgraph_data_type_id
    rocgraph_type_erased_host_array_type(const rocgraph_type_erased_host_array_view_t* p);

/**
 * @brief    Get the raw pointer of the type erased host array view
 *
 * @param [in]  p    Pointer to the type erased host array view
 * @return Pointer (host memory) for the data in the array
 */
ROCGRAPH_EXPORT void*
    rocgraph_type_erased_host_array_pointer(const rocgraph_type_erased_host_array_view_t* p);

/**
 * @brief    Copy data between two type erased device array views
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [out] dst         Pointer to type erased host array view destination
 * @param [in]  src         Pointer to type erased host array view source
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_type_erased_host_array_view_copy(const rocgraph_handle_t*                      handle,
                                              rocgraph_type_erased_host_array_view_t*       dst,
                                              const rocgraph_type_erased_host_array_view_t* src,
                                              rocgraph_error_t**                            error);

#ifdef __cplusplus
}
#endif
