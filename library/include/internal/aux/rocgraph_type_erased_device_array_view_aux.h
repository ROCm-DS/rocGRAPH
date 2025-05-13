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
#include "internal/types/rocgraph_type_erased_device_array_view_t.h"
#include "rocgraph-export.h"

/** @ingroup aux_module
 */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief    Copy data from host to a type erased device array view
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [out] dst         Pointer to the type erased device array view
 * @param [in]  h_src       Pointer to host array to copy into device memory
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status rocgraph_type_erased_device_array_view_copy_from_host(
    const rocgraph_handle_t*                  handle,
    rocgraph_type_erased_device_array_view_t* dst,
    const rocgraph_byte_t*                    h_src,
    rocgraph_error_t**                        error);

/**
 * @brief    Copy data from device to a type erased host array
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [out] h_dst       Pointer to host array
 * @param [in]  src         Pointer to the type erased device array view source
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status rocgraph_type_erased_device_array_view_copy_to_host(
    const rocgraph_handle_t*                        handle,
    rocgraph_byte_t*                                h_dst,
    const rocgraph_type_erased_device_array_view_t* src,
    rocgraph_error_t**                              error);

/**
 * @brief    Copy data between two type erased device array views
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [out] dst         Pointer to type erased device array view destination
 * @param [in]  src         Pointer to type erased device array view source
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_type_erased_device_array_view_copy(const rocgraph_handle_t*                  handle,
                                                rocgraph_type_erased_device_array_view_t* dst,
                                                const rocgraph_type_erased_device_array_view_t* src,
                                                rocgraph_error_t** error);

/**
 * @brief Create a type erased device array view with a different type
 *
 *    Create a type erased device array view from
 *    a type erased device array treating the underlying
 *    pointer as a different type.
 *
 *    Note: This is only viable when the underlying types are the same size.  That
 *    is, you can switch between INT32 and FLOAT32, or between INT64 and FLOAT64.
 *    But if the types are different sizes this will be an error.
 *
 * @param [in]  array        Pointer to the type erased device array
 * @param [in]  dtype        The type to cast the pointer to
 * @param [out] result_view  Address where to put the allocated device view
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status rocgraph_type_erased_device_array_view_as_type(
    rocgraph_type_erased_device_array_t*       array,
    rocgraph_data_type_id                      dtype,
    rocgraph_type_erased_device_array_view_t** result_view,
    rocgraph_error_t**                         error);

/**
 * @brief    Create a type erased device array view from
 *           a raw device pointer.
 *
 * @param [in]  pointer     Raw device pointer
 * @param [in]  n_elems     The number of elements in the array
 * @param [in]  dtype       The type of array to create
 * @return Pointer to the view of the host array
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_type_erased_device_array_view_create(void*                 pointer,
                                                  size_t                n_elems,
                                                  rocgraph_data_type_id dtype);

/**
 * @brief    Destroy a type erased device array view
 *
 * @param [in]  p    Pointer to the type erased device array view
 */
ROCGRAPH_EXPORT void
    rocgraph_type_erased_device_array_view_free(rocgraph_type_erased_device_array_view_t* p);

/**
 * @brief    Get the size of a type erased device array view
 *
 * @param [in]  p    Pointer to the type erased device array view
 * @return The number of elements in the array
 */
ROCGRAPH_EXPORT size_t
    rocgraph_type_erased_device_array_view_size(const rocgraph_type_erased_device_array_view_t* p);

/**
 * @brief    Get the type of a type erased device array view
 *
 * @param [in]  p    Pointer to the type erased device array view
 * @return The type of the elements in the array
 */
ROCGRAPH_EXPORT rocgraph_data_type_id
    rocgraph_type_erased_device_array_view_type(const rocgraph_type_erased_device_array_view_t* p);

/**
 * @brief    Get the raw pointer of the type erased device array view
 *
 * @param [in]  p    Pointer to the type erased device array view
 * @return Pointer (device memory) for the data in the array
 */
ROCGRAPH_EXPORT const void* rocgraph_type_erased_device_array_view_pointer(
    const rocgraph_type_erased_device_array_view_t* p);

#ifdef __cplusplus
}
#endif
