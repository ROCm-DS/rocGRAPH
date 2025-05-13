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
#include "internal/types/rocgraph_degrees_result_t.h"
#include "internal/types/rocgraph_error_t.h"
#include "internal/types/rocgraph_graph_t.h"
#include "internal/types/rocgraph_handle_t.h"
#include "internal/types/rocgraph_induced_subgraph_result_t.h"
#include "internal/types/rocgraph_status.h"
#include "internal/types/rocgraph_type_erased_device_array_view_t.h"
#include "internal/types/rocgraph_vertex_pairs_t.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief       Create vertex_pairs
 *
 * Input data will be shuffled to the proper GPU and stored in the
 * output vertex_pairs.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Graph to operate on
 * @param [in]  first        Type erased array of vertex ids for the first vertex of the pair
 * @param [in]  second       Type erased array of vertex ids for the second vertex of the pair
 * @param [out] vertex_pairs Opaque pointer to vertex_pairs
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_create_vertex_pairs(const rocgraph_handle_t*                        handle,
                                 rocgraph_graph_t*                               graph,
                                 const rocgraph_type_erased_device_array_view_t* first,
                                 const rocgraph_type_erased_device_array_view_t* second,
                                 rocgraph_vertex_pairs_t**                       vertex_pairs,
                                 rocgraph_error_t**                              error);

/**
 * @brief      Find all 2-hop neighbors in the graph
 *
 * @param [in]  handle         Handle for accessing resources
 * @param [in]  graph          Pointer to graph
 * @param [in]  start_vertices Optional type erased array of starting vertices
 *                             If NULL use all, if specified compute two-hop
 *                             neighbors for these starting vertices
 * @param [in]  do_expensive_check
 *                             A flag to run expensive checks for input arguments (if set to true)
 * @param [out] result         Opaque pointer to resulting vertex pairs
 * @param [out] error          Pointer to an error object storing details of any error.  Will
 *                             be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_two_hop_neighbors(const rocgraph_handle_t*                        handle,
                               rocgraph_graph_t*                               graph,
                               const rocgraph_type_erased_device_array_view_t* start_vertices,
                               rocgraph_bool                                   do_expensive_check,
                               rocgraph_vertex_pairs_t**                       result,
                               rocgraph_error_t**                              error);

/**
 * @brief      Extract induced subgraph(s)
 *
 * Given a list of vertex ids, extract a list of edges that represent the subgraph
 * containing only the specified vertex ids.
 *
 * This function will do multiple subgraph extractions concurrently.  The vertex ids
 * are specified in CSR-style, with @p subgraph_vertices being a list of vertex ids
 * and @p subgraph_offsets[i] identifying the start offset for each extracted subgraph
 *
 * @param [in]  handle            Handle for accessing resources
 * @param [in]  graph             Pointer to graph
 * @param [in]  subgraph_offsets  Type erased array of subgraph offsets into
 *                                @p subgraph_vertices
 * @param [in]  subgraph_vertices Type erased array of vertices to include in
 *                                extracted subgraph.
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result            Opaque pointer to induced subgraph result
 * @param [out] error             Pointer to an error object storing details of any error.  Will
 *                                be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status rocgraph_extract_induced_subgraph(
    const rocgraph_handle_t*                        handle,
    rocgraph_graph_t*                               graph,
    const rocgraph_type_erased_device_array_view_t* subgraph_offsets,
    const rocgraph_type_erased_device_array_view_t* subgraph_vertices,
    rocgraph_bool                                   do_expensive_check,
    rocgraph_induced_subgraph_result_t**            result,
    rocgraph_error_t**                              error);

/**
 * @brief      Gather edgelist
 *
 * This function collects the edgelist from all ranks and stores the combine edgelist
 * in each rank
 *
 * @param [in]  handle            Handle for accessing resources.
 * @param [in]  src               Device array containing the source vertex ids.
 * @param [in]  dst               Device array containing the destination vertex ids
 * @param [in]  weights           Optional device array containing the edge weights
 * @param [in]  edge_ids          Optional device array containing the edge ids for each edge.
 * @param [in]  edge_type_ids     Optional device array containing the edge types for each edge
 * @param [out] result            Opaque pointer to gathered edgelist result
 * @param [out] error             Pointer to an error object storing details of any error.  Will
 *                                be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_allgather(const rocgraph_handle_t*                        handle,
                       const rocgraph_type_erased_device_array_view_t* src,
                       const rocgraph_type_erased_device_array_view_t* dst,
                       const rocgraph_type_erased_device_array_view_t* weights,
                       const rocgraph_type_erased_device_array_view_t* edge_ids,
                       const rocgraph_type_erased_device_array_view_t* edge_type_ids,
                       rocgraph_induced_subgraph_result_t**            result,
                       rocgraph_error_t**                              error);

/**
 * @brief      Count multi_edges
 *
 * Count the number of multi-edges in the graph
 *
 * @param [in]  handle              Handle for accessing resources.
 * @param [in]  graph               Pointer to graph
 * @param [in]  do_expensive_check  A flag to run expensive checks for input arguments (if set to
 * true)
 * @param [out] result              Where to store the count of multi-edges
 * @param [out] error               Pointer to an error object storing details of any error.  Will
 *                                  be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status rocgraph_count_multi_edges(const rocgraph_handle_t* handle,
                                                           rocgraph_graph_t*        graph,
                                                           rocgraph_bool      do_expensive_check,
                                                           size_t*            result,
                                                           rocgraph_error_t** error);

/**
 * @brief      Compute in degrees
 *
 * Compute the in degrees for the vertices in the graph.
 *
 * @param [in]  handle              Handle for accessing resources.
 * @param [in]  graph               Pointer to graph
 * @param [in]  source_vertices     Device array of vertices we want to compute in degrees for.
 * @param [in]  do_expensive_check  A flag to run expensive checks for input arguments (if set to
 * true)
 * @param [out] result              Opaque pointer to degrees result
 * @param [out] error               Pointer to an error object storing details of any error.  Will
 *                                  be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_in_degrees(const rocgraph_handle_t*                        handle,
                        rocgraph_graph_t*                               graph,
                        const rocgraph_type_erased_device_array_view_t* source_vertices,
                        rocgraph_bool                                   do_expensive_check,
                        rocgraph_degrees_result_t**                     result,
                        rocgraph_error_t**                              error);

/**
 * @brief      Compute out degrees
 *
 * Compute the out degrees for the vertices in the graph.
 *
 * @param [in]  handle              Handle for accessing resources.
 * @param [in]  graph               Pointer to graph
 * @param [in]  source_vertices     Device array of vertices we want to compute out degrees for.
 * @param [in]  do_expensive_check  A flag to run expensive checks for input arguments (if set to
 * true)
 * @param [out] result              Opaque pointer to degrees result
 * @param [out] error               Pointer to an error object storing details of any error.  Will
 *                                  be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_out_degrees(const rocgraph_handle_t*                        handle,
                         rocgraph_graph_t*                               graph,
                         const rocgraph_type_erased_device_array_view_t* source_vertices,
                         rocgraph_bool                                   do_expensive_check,
                         rocgraph_degrees_result_t**                     result,
                         rocgraph_error_t**                              error);

/**
 * @brief      Compute degrees
 *
 * Compute the degrees for the vertices in the graph.
 *
 * @param [in]  handle              Handle for accessing resources.
 * @param [in]  graph               Pointer to graph
 * @param [in]  source_vertices     Device array of vertices we want to compute degrees for.
 * @param [in]  do_expensive_check  A flag to run expensive checks for input arguments (if set to
 * true)
 * @param [out] result              Opaque pointer to degrees result
 * @param [out] error               Pointer to an error object storing details of any error.  Will
 *                                  be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_degrees(const rocgraph_handle_t*                        handle,
                     rocgraph_graph_t*                               graph,
                     const rocgraph_type_erased_device_array_view_t* source_vertices,
                     rocgraph_bool                                   do_expensive_check,
                     rocgraph_degrees_result_t**                     result,
                     rocgraph_error_t**                              error);

#ifdef __cplusplus
}
#endif
