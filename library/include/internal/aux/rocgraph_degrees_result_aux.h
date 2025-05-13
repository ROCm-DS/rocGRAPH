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

#include "internal/types/rocgraph_degrees_result_t.h"
#include "internal/types/rocgraph_type_erased_device_array_view_t.h"
#include "rocgraph-export.h"

/** @ingroup aux_module
 */
#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief       Get the vertex ids
 *
 * @param [in]     degrees_result   Opaque pointer to degree result
 * @return type erased array view of vertex ids
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_degrees_result_get_vertices(rocgraph_degrees_result_t* degrees_result);

/**
 * @brief       Get the in degrees
 *
 * @param [in]     degrees_result   Opaque pointer to degree result
 * @return type erased array view of vertex ids
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_degrees_result_get_in_degrees(rocgraph_degrees_result_t* degrees_result);

/**
 * @brief       Get the out degrees
 *
 * If the graph is symmetric, in degrees and out degrees will be equal (and
 * will be stored in the same memory).
 *
 * @param [in]     degrees_result   Opaque pointer to degree result
 * @return type erased array view of vertex ids
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_degrees_result_get_out_degrees(rocgraph_degrees_result_t* degrees_result);

/**
 * @brief     Free degree result
 *
 * @param [in]    degrees_result   Opaque pointer to degree result
 */
ROCGRAPH_EXPORT void rocgraph_degrees_result_free(rocgraph_degrees_result_t* degrees_result);

#ifdef __cplusplus
}
#endif
