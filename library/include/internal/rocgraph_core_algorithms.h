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

#include "rocgraph-export.h"

#include "internal/types/rocgraph_bool.h"
#include "internal/types/rocgraph_core_result_t.h"
#include "internal/types/rocgraph_error_t.h"
#include "internal/types/rocgraph_graph_t.h"
#include "internal/types/rocgraph_handle_t.h"
#include "internal/types/rocgraph_k_core_degree_type.h"
#include "internal/types/rocgraph_k_core_result_t.h"
#include "internal/types/rocgraph_status.h"

/** @defgroup core Core algorithms
 */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief     Perform core number.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * @param [in]  degree_type  Compute core_number using in, out or both in and out edges
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result       Opaque pointer to core number results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status rocgraph_core_number(const rocgraph_handle_t*    handle,
                                                     rocgraph_graph_t*           graph,
                                                     rocgraph_k_core_degree_type degree_type,
                                                     rocgraph_bool               do_expensive_check,
                                                     rocgraph_core_result_t**    result,
                                                     rocgraph_error_t**          error);

/**
 * @brief     Perform k_core using output from core_number
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * @param [in]  k            The value of k to use
 * @param [in]  degree_type  Compute core_number using in, out or both in and out edges.
 *                           Ignored if core_result is specified.
 * @param [in]  core_result  Result from calling rocgraph_core_number, if NULL then
 *                           call core_number inside this function call.
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result       Opaque pointer to k_core results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status rocgraph_k_core(const rocgraph_handle_t*      handle,
                                                rocgraph_graph_t*             graph,
                                                size_t                        k,
                                                rocgraph_k_core_degree_type   degree_type,
                                                const rocgraph_core_result_t* core_result,
                                                rocgraph_bool                 do_expensive_check,
                                                rocgraph_k_core_result_t**    result,
                                                rocgraph_error_t**            error);

#ifdef __cplusplus
}
#endif
