// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*!\file*/
/*
 * Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 */
/*
 * Copyright (C) 2022-2024, NVIDIA CORPORATION.
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

#include "internal/types/rocgraph_core_result_t.h"
#include "internal/types/rocgraph_error_t.h"
#include "internal/types/rocgraph_handle_t.h"
#include "internal/types/rocgraph_status.h"
#include "internal/types/rocgraph_type_erased_device_array_view_t.h"
#include "rocgraph-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup core
 * @brief       Create a core_number result (in case it was previously extracted)
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  vertices     The result from core number
 * @param [in]  core_numbers The result from core number
 * @param [out] core_result       Opaque pointer to core number results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_core_result_create(const rocgraph_handle_t*                  handle,
                                rocgraph_type_erased_device_array_view_t* vertices,
                                rocgraph_type_erased_device_array_view_t* core_numbers,
                                rocgraph_core_result_t**                  core_result,
                                rocgraph_error_t**                        error);

/**
 * @ingroup core
 * @brief       Get the vertex ids from the core result
 *
 * @param [in]     result   The result from core number
 * @return type erased array of vertex ids
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_core_result_get_vertices(rocgraph_core_result_t* result);

/**
 * @ingroup core
 * @brief       Get the core numbers from the core result
 *
 * @param [in]    result    The result from core number
 * @return type erased array of core numbers
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_core_result_get_core_numbers(rocgraph_core_result_t* result);

/**
 * @ingroup core
 * @brief     Free core result
 *
 * @param [in]    result    The result from core number
 */
ROCGRAPH_EXPORT void rocgraph_core_result_free(rocgraph_core_result_t* result);

#ifdef __cplusplus
}
#endif
