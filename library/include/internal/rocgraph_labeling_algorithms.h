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
#include "internal/types/rocgraph_error_t.h"
#include "internal/types/rocgraph_graph_t.h"
#include "internal/types/rocgraph_handle_t.h"
#include "internal/types/rocgraph_labeling_result_t.h"
#include "internal/types/rocgraph_status.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup labeling Labeling algorithms
 */
/**
 * @brief Labels each vertex in the input graph with its (weakly-connected-)component ID
 *
 * The input graph must be symmetric. Component IDs can be arbitrary integers (they can be
 * non-consecutive and are not ordered by component size or any other criterion).
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  graph       Pointer to graph
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result      Opaque pointer to labeling results
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not ROCGRAPH_SUCCESS
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_weakly_connected_components(const rocgraph_handle_t*     handle,
                                         rocgraph_graph_t*            graph,
                                         rocgraph_bool                do_expensive_check,
                                         rocgraph_labeling_result_t** result,
                                         rocgraph_error_t**           error);

/**
 * @brief Labels each vertex in the input graph with its (strongly-connected-)component ID
 *
 * The input graph may be asymmetric. Component IDs can be arbitrary integers (they can be
 * non-consecutive and are not ordered by component size or any other criterion).
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  graph       Pointer to graph
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result      Opaque pointer to labeling results
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not ROCGRAPH_SUCCESS
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_strongly_connected_components(const rocgraph_handle_t*     handle,
                                           rocgraph_graph_t*            graph,
                                           rocgraph_bool                do_expensive_check,
                                           rocgraph_labeling_result_t** result,
                                           rocgraph_error_t**           error);

#ifdef __cplusplus
}
#endif
