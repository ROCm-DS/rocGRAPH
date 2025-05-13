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

#include "internal/types/rocgraph_bool.h"
#include "internal/types/rocgraph_error_t.h"
#include "internal/types/rocgraph_extract_paths_result_t.h"
#include "internal/types/rocgraph_graph_t.h"
#include "internal/types/rocgraph_handle_t.h"
#include "internal/types/rocgraph_paths_result_t.h"
#include "internal/types/rocgraph_status.h"
#include "internal/types/rocgraph_type_erased_device_array_view_t.h"
#include "rocgraph-export.h"

/** @defgroup traversal Traversal Algorithms
 *  @ingroup c_api
 */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief     Perform a breadth first search from a set of seed vertices.
 *
 * This function computes the distances (minimum number of hops to reach the vertex) from the source
 * vertex. If @p predecessors is not NULL, this function calculates the predecessor of each
 * vertex (parent vertex in the breadth-first search tree) as well.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * FIXME:  Make this just [in], copy it if I need to temporarily modify internally
 * @param [in,out]  sources  Array of source vertices.  NOTE: Array might be modified if
 *                           renumbering is enabled for the graph
 * @param [in]  direction_optimizing If set to true, this algorithm switches between the push based
 * breadth-first search and pull based breadth-first search depending on the size of the
 * breadth-first search frontier (currently unsupported). This option is valid only for symmetric
 * input graphs.
 * @param depth_limit Sets the maximum number of breadth-first search iterations. Any vertices
 * farther than @p depth_limit hops from @p source_vertex will be marked as unreachable.
 * @param [in] compute_predecessors A flag to indicate whether to compute the predecessors in the
 * result
 * @param [in] do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result       Opaque pointer to paths results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status rocgraph_bfs(const rocgraph_handle_t*                  handle,
                                             rocgraph_graph_t*                         graph,
                                             rocgraph_type_erased_device_array_view_t* sources,
                                             rocgraph_bool             direction_optimizing,
                                             size_t                    depth_limit,
                                             rocgraph_bool             compute_predecessors,
                                             rocgraph_bool             do_expensive_check,
                                             rocgraph_paths_result_t** result,
                                             rocgraph_error_t**        error);

/**
 * @brief     Perform single-source shortest-path to compute the minimum distances
 *            (and predecessors) from the source vertex.
 *
 * This function computes the distances (minimum edge weight sums) from the source
 * vertex. If @p predecessors is not NULL, this function calculates the predecessor of each
 * vertex (parent vertex in the breadth-first search tree) as well.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * @param [in]  source       Source vertex id
 * @param [in]  cutoff       Maximum edge weight sum to consider
 * @param [in]  compute_predecessors A flag to indicate whether to compute the predecessors in the
 * result
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result       Opaque pointer to paths results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status rocgraph_sssp(const rocgraph_handle_t*  handle,
                                              rocgraph_graph_t*         graph,
                                              size_t                    source,
                                              double                    cutoff,
                                              rocgraph_bool             compute_predecessors,
                                              rocgraph_bool             do_expensive_check,
                                              rocgraph_paths_result_t** result,
                                              rocgraph_error_t**        error);

/**
 * @brief     Extract BFS or SSSP paths from a rocgraph_paths_result_t
 *
 * This function extracts paths from the BFS or SSSP output.  BFS and SSSP output
 * distances and predecessors.  The path from a vertex v back to the original
 * source vertex can be extracted by recursively looking up the predecessor
 * vertex until you arrive back at the original source vertex.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [in]  sources      Array of source vertices
 * @param [in]  paths_result Output from the BFS call
 * @param [in]  destinations Array of destination vertices.
 * @param [out] result       Opaque pointer to extract_paths results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_extract_paths(const rocgraph_handle_t*                        handle,
                           rocgraph_graph_t*                               graph,
                           const rocgraph_type_erased_device_array_view_t* sources,
                           const rocgraph_paths_result_t*                  paths_result,
                           const rocgraph_type_erased_device_array_view_t* destinations,
                           rocgraph_extract_paths_result_t**               result,
                           rocgraph_error_t**                              error);

#ifdef __cplusplus
}
#endif
