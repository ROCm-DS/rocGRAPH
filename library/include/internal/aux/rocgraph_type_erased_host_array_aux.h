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
 * @brief     Create a type erased host array
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  n_elems     The number of elements in the array
 * @param [in]  dtype       The type of array to create
 * @param [out] array       Pointer to the location to store the pointer to the host array
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_type_erased_host_array_create(const rocgraph_handle_t*            handle,
                                           size_t                              n_elems,
                                           rocgraph_data_type_id               dtype,
                                           rocgraph_type_erased_host_array_t** array,
                                           rocgraph_error_t**                  error);

/**
 * @brief    Destroy a type erased host array
 *
 * @param [in]  p    Pointer to the type erased host array
 */
ROCGRAPH_EXPORT void rocgraph_type_erased_host_array_free(rocgraph_type_erased_host_array_t* p);

/**
 * @brief    Create a type erased host array view from
 *           a type erased host array
 *
 * @param [in]  array       Pointer to the type erased host array
 * @return Pointer to the view of the host array
 */
ROCGRAPH_EXPORT rocgraph_type_erased_host_array_view_t*
    rocgraph_type_erased_host_array_view(rocgraph_type_erased_host_array_t* array);

#ifdef __cplusplus
}
#endif
