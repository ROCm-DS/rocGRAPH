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
#include "internal/types/rocgraph_compression_type.h"
#include "internal/types/rocgraph_error_t.h"
#include "internal/types/rocgraph_graph_t.h"
#include "internal/types/rocgraph_handle_t.h"
#include "internal/types/rocgraph_prior_sources_behavior.h"
#include "internal/types/rocgraph_random_walk_result_t.h"
#include "internal/types/rocgraph_rng_state_t.h"
#include "internal/types/rocgraph_sample_result_t.h"
#include "internal/types/rocgraph_sampling_options_t.h"
#include "internal/types/rocgraph_status.h"
#include "internal/types/rocgraph_type_erased_device_array_t.h"
#include "internal/types/rocgraph_type_erased_device_array_view_t.h"
#include "internal/types/rocgraph_type_erased_host_array_view_t.h"
#include "rocgraph-export.h"

/** @defgroup samplingC Sampling algorithms
 */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Compute uniform random walks
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  start_vertices  Array of source vertices
 * @param [in]  max_length      Maximum length of the generated path
 * @param [in]  result          Output from the node2vec call
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_uniform_random_walks(const rocgraph_handle_t*                        handle,
                                  rocgraph_graph_t*                               graph,
                                  const rocgraph_type_erased_device_array_view_t* start_vertices,
                                  size_t                                          max_length,
                                  rocgraph_random_walk_result_t**                 result,
                                  rocgraph_error_t**                              error);

/**
 * @brief  Compute biased random walks
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  start_vertices  Array of source vertices
 * @param [in]  max_length      Maximum length of the generated path
 * @param [in]  result          Output from the node2vec call
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_biased_random_walks(const rocgraph_handle_t*                        handle,
                                 rocgraph_graph_t*                               graph,
                                 const rocgraph_type_erased_device_array_view_t* start_vertices,
                                 size_t                                          max_length,
                                 rocgraph_random_walk_result_t**                 result,
                                 rocgraph_error_t**                              error);

/**
 * @brief  Compute random walks using the node2vec framework.
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  start_vertices  Array of source vertices
 * @param [in]  max_length      Maximum length of the generated path
 * @param [in]  p               The return parameter
 * @param [in]  q               The in/out parameter
 * @param [in]  result          Output from the node2vec call
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_node2vec_random_walks(const rocgraph_handle_t*                        handle,
                                   rocgraph_graph_t*                               graph,
                                   const rocgraph_type_erased_device_array_view_t* start_vertices,
                                   size_t                                          max_length,
                                   double                                          p,
                                   double                                          q,
                                   rocgraph_random_walk_result_t**                 result,
                                   rocgraph_error_t**                              error);

/**
 * @brief  Compute random walks using the node2vec framework.
 * @deprecated This call should be replaced with rocgraph_node2vec_random_walks
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [in]  sources      Array of source vertices
 * @param [in]  max_depth    Maximum length of the generated path
 * @param [in]  compress_result If true, return the paths as a compressed sparse row matrix,
 *                              otherwise return as a dense matrix
 * @param [in]  p            The return parameter
 * @param [in]  q            The in/out parameter
 * @param [in]  result       Output from the node2vec call
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_node2vec(const rocgraph_handle_t*                        handle,
                      rocgraph_graph_t*                               graph,
                      const rocgraph_type_erased_device_array_view_t* sources,
                      size_t                                          max_depth,
                      rocgraph_bool                                   compress_result,
                      double                                          p,
                      double                                          q,
                      rocgraph_random_walk_result_t**                 result,
                      rocgraph_error_t**                              error);

/**
 * @brief     Uniform Neighborhood Sampling
 *
 * Returns a sample of the neighborhood around specified start vertices.  Optionally, each
 * start vertex can be associated with a label, allowing the caller to specify multiple batches
 * of sampling requests in the same function call - which should improve GPU utilization.
 *
 * If label is NULL then all start vertices will be considered part of the same batch and the
 * return value will not have a label column.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [in]  start_vertices Device array of start vertices for the sampling
 * @param [in]  start_vertex_labels  Device array of start vertex labels for the sampling.  The
 * labels associated with each start vertex will be included in the output associated with results
 * that were derived from that start vertex.  We only support label of type INT32. If label is
 * NULL, the return data will not be labeled.
 * @param [in]  label_list Device array of the labels included in @p start_vertex_labels.  If
 * @p label_to_comm_rank is not specified this parameter is ignored.  If specified, label_list
 * must be sorted in ascending order.
 * @param [in]  label_to_comm_rank Device array identifying which comm rank the output for a
 * particular label should be shuffled in the output.  If not specifed the data is not organized in
 * output.  If specified then the all data from @p label_list[i] will be shuffled to rank @p.  This
 * cannot be specified unless @p start_vertex_labels is also specified
 * label_to_comm_rank[i].  If not specified then the output data will not be shuffled between ranks.
 * @param [in]  label_offsets Device array of the offsets for each label in the seed list.  This
 *                            parameter is only used with the retain_seeds option.
 * @param [in]  fan_out       Host array defining the fan out at each step in the sampling algorithm.
 *                           We only support fanout values of type INT32
 * @param [in,out] rng_state State of the random number generator, updated with each call
 * @param [in]  options
 *                           Opaque pointer defining the sampling options.
 * @param [in]  do_expensive_check
 *                           A flag to run expensive checks for input arguments (if set to true)
 * @param [in]  result       Output from the uniform_neighbor_sample call
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status rocgraph_uniform_neighbor_sample(
    const rocgraph_handle_t*                        handle,
    rocgraph_graph_t*                               graph,
    const rocgraph_type_erased_device_array_view_t* start_vertices,
    const rocgraph_type_erased_device_array_view_t* start_vertex_labels,
    const rocgraph_type_erased_device_array_view_t* label_list,
    const rocgraph_type_erased_device_array_view_t* label_to_comm_rank,
    const rocgraph_type_erased_device_array_view_t* label_offsets,
    const rocgraph_type_erased_host_array_view_t*   fan_out,
    rocgraph_rng_state_t*                           rng_state,
    const rocgraph_sampling_options_t*              options,
    rocgraph_bool                                   do_expensive_check,
    rocgraph_sample_result_t**                      result,
    rocgraph_error_t**                              error);

/**
 * @ingroup samplingC
 * @brief Select random vertices from the graph
 *
 * @param [in]      handle        Handle for accessing resources
 * @param [in]      graph         Pointer to graph
 * @param [in,out]  rng_state     State of the random number generator, updated with each call
 * @param [in]      num_vertices  Number of vertices to sample
 * @param [out]     vertices      Device array view to populate label
 * @param [out]     error         Pointer to an error object storing details of
 *                                any error.  Will be populated if error code is
 *                                not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_select_random_vertices(const rocgraph_handle_t*              handle,
                                    const rocgraph_graph_t*               graph,
                                    rocgraph_rng_state_t*                 rng_state,
                                    size_t                                num_vertices,
                                    rocgraph_type_erased_device_array_t** vertices,
                                    rocgraph_error_t**                    error);

#ifdef __cplusplus
}
#endif
