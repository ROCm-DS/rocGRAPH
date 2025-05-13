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
#include "internal/types/rocgraph_centrality_result_t.h"
#include "internal/types/rocgraph_edge_centrality_result_t.h"
#include "internal/types/rocgraph_error_t.h"
#include "internal/types/rocgraph_graph_t.h"
#include "internal/types/rocgraph_handle_t.h"
#include "internal/types/rocgraph_hits_result_t.h"
#include "internal/types/rocgraph_status.h"
#include "internal/types/rocgraph_type_erased_device_array_view_t.h"

/** @defgroup centrality Centrality algorithms
 */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief     Compute pagerank
 *
 * @param [in]  handle      Handle for accessing resources.
 * @param [in]  graph       Pointer to graph.
 * @param [in]  precomputed_vertex_out_weight_vertices
 *                          Optionally send in precomputed sum of vertex out weights
 *                          (a performance optimization).  This defines the vertices.
 *                          Set to NULL if no value is passed.
 * @param [in]  precomputed_vertex_out_weight_sums
 *                          Optionally send in precomputed sum of vertex out weights
 *                          (a performance optimization).  Set to NULL if
 *                          no value is passed.
 * @param [in]  initial_guess_vertices
 *                          Optionally send in an initial guess of the pagerank values
 *                          (a performance optimization).  This defines the vertices.
 *                          Set to NULL if no value is passed. If NULL, initial PageRank
 *                          values are set to 1.0 divided by the number of vertices in
 *                          the graph.
 * @param [in]  initial_guess_values
 *                          Optionally send in an initial guess of the pagerank values
 *                          (a performance optimization).  Set to NULL if
 *                          no value is passed. If NULL, initial PageRank values are set
 *                          to 1.0 divided by the number of vertices in the graph.
 * @param [in]  alpha       PageRank damping factor.
 * @param [in]  epsilon     Error tolerance to check convergence. Convergence is assumed
 *                          if the sum of the differences in PageRank values between two
 *                          consecutive iterations is less than the number of vertices
 *                          in the graph multiplied by @p epsilon.
 * @param [in]  max_iterations Maximum number of PageRank iterations.
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result      Opaque pointer to pagerank results.
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */

ROCGRAPH_EXPORT rocgraph_status rocgraph_pagerank(
    const rocgraph_handle_t*                        handle,
    rocgraph_graph_t*                               graph,
    const rocgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices,
    const rocgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
    const rocgraph_type_erased_device_array_view_t* initial_guess_vertices,
    const rocgraph_type_erased_device_array_view_t* initial_guess_values,
    double                                          alpha,
    double                                          epsilon,
    size_t                                          max_iterations,
    rocgraph_bool                                   do_expensive_check,
    rocgraph_centrality_result_t**                  result,
    rocgraph_error_t**                              error);

/**
 * @brief     Compute pagerank
 *
 * @deprecated This version of pagerank should be dropped in favor
 *             of the rocgraph_pagerank_allow_nonconvergence version.
 *             Eventually that version will be renamed to this version.
 *
 * @param [in]  handle      Handle for accessing resources.
 * @param [in]  graph       Pointer to graph.
 * @param [in]  precomputed_vertex_out_weight_vertices
 *                          Optionally send in precomputed sum of vertex out weights
 *                          (a performance optimization).  This defines the vertices.
 *                          Set to NULL if no value is passed.
 * @param [in]  precomputed_vertex_out_weight_sums
 *                          Optionally send in precomputed sum of vertex out weights
 *                          (a performance optimization).  Set to NULL if
 *                          no value is passed.
 * @param [in]  initial_guess_vertices
 *                          Optionally send in an initial guess of the pagerank values
 *                          (a performance optimization).  This defines the vertices.
 *                          Set to NULL if no value is passed. If NULL, initial PageRank
 *                          values are set to 1.0 divided by the number of vertices in
 *                          the graph.
 * @param [in]  initial_guess_values
 *                          Optionally send in an initial guess of the pagerank values
 *                          (a performance optimization).  Set to NULL if
 *                          no value is passed. If NULL, initial PageRank values are set
 *                          to 1.0 divided by the number of vertices in the graph.
 * @param [in]  alpha       PageRank damping factor.
 * @param [in]  epsilon     Error tolerance to check convergence. Convergence is assumed
 *                          if the sum of the differences in PageRank values between two
 *                          consecutive iterations is less than the number of vertices
 *                          in the graph multiplied by @p epsilon.
 * @param [in]  max_iterations Maximum number of PageRank iterations.
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result      Opaque pointer to pagerank results.
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not ROCGRAPH_SUCCESS.
 *
 * @return error            Returned error code.
 */
ROCGRAPH_EXPORT rocgraph_status rocgraph_pagerank_allow_nonconvergence(
    const rocgraph_handle_t*                        handle,
    rocgraph_graph_t*                               graph,
    const rocgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices,
    const rocgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
    const rocgraph_type_erased_device_array_view_t* initial_guess_vertices,
    const rocgraph_type_erased_device_array_view_t* initial_guess_values,
    double                                          alpha,
    double                                          epsilon,
    size_t                                          max_iterations,
    rocgraph_bool                                   do_expensive_check,
    rocgraph_centrality_result_t**                  result,
    rocgraph_error_t**                              error);

/**
 * @brief     Compute personalized pagerank
 *
 * @deprecated This version of personalized pagerank should be dropped in favor
 *             of the rocgraph_personalized_pagerank_allow_nonconvergence version.
 *             Eventually that version will be renamed to this version.
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  graph       Pointer to graph
 * @param [in]  precomputed_vertex_out_weight_vertices
 *                          Optionally send in precomputed sum of vertex out weights
 *                          (a performance optimization).  This defines the vertices.
 *                          Set to NULL if no value is passed.
 * @param [in]  precomputed_vertex_out_weight_sums
 *                          Optionally send in precomputed sum of vertex out weights
 *                          (a performance optimization).  Set to NULL if
 *                          no value is passed.
 * @param [in]  initial_guess_vertices
 *                          Optionally send in an initial guess of the pagerank values
 *                          (a performance optimization).  This defines the vertices.
 *                          Set to NULL if no value is passed. If NULL, initial PageRank
 *                          values are set to 1.0 divided by the number of vertices in
 *                          the graph.
 * @param [in]  initial_guess_values
 *                          Optionally send in an initial guess of the pagerank values
 *                          (a performance optimization).  Set to NULL if
 *                          no value is passed. If NULL, initial PageRank values are set
 *                          to 1.0 divided by the number of vertices in the graph.
 * @param [in]  personalization_vertices Pointer to an array storing personalization vertex
 * identifiers (compute personalized PageRank).
 * @param [in]  personalization_values Pointer to an array storing personalization values for the
 * vertices in the personalization set.
 * @param [in]  alpha       PageRank damping factor.
 * @param [in]  epsilon     Error tolerance to check convergence. Convergence is assumed
 *                          if the sum of the differences in PageRank values between two
 *                          consecutive iterations is less than the number of vertices
 *                          in the graph multiplied by @p epsilon.
 * @param [in]  max_iterations Maximum number of PageRank iterations.
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result      Opaque pointer to pagerank results
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status rocgraph_personalized_pagerank(
    const rocgraph_handle_t*                        handle,
    rocgraph_graph_t*                               graph,
    const rocgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices,
    const rocgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
    const rocgraph_type_erased_device_array_view_t* initial_guess_vertices,
    const rocgraph_type_erased_device_array_view_t* initial_guess_values,
    const rocgraph_type_erased_device_array_view_t* personalization_vertices,
    const rocgraph_type_erased_device_array_view_t* personalization_values,
    double                                          alpha,
    double                                          epsilon,
    size_t                                          max_iterations,
    rocgraph_bool                                   do_expensive_check,
    rocgraph_centrality_result_t**                  result,
    rocgraph_error_t**                              error);

/**
 * @brief     Compute personalized pagerank
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  graph       Pointer to graph
 * @param [in]  precomputed_vertex_out_weight_vertices
 *                          Optionally send in precomputed sum of vertex out weights
 *                          (a performance optimization).  This defines the vertices.
 *                          Set to NULL if no value is passed.
 * @param [in]  precomputed_vertex_out_weight_sums
 *                          Optionally send in precomputed sum of vertex out weights
 *                          (a performance optimization).  Set to NULL if
 *                          no value is passed.
 * @param [in]  initial_guess_vertices
 *                          Optionally send in an initial guess of the pagerank values
 *                          (a performance optimization).  This defines the vertices.
 *                          Set to NULL if no value is passed. If NULL, initial PageRank
 *                          values are set to 1.0 divided by the number of vertices in
 *                          the graph.
 * @param [in]  initial_guess_values
 *                          Optionally send in an initial guess of the pagerank values
 *                          (a performance optimization).  Set to NULL if
 *                          no value is passed. If NULL, initial PageRank values are set
 *                          to 1.0 divided by the number of vertices in the graph.
 * @param [in]  personalization_vertices Pointer to an array storing personalization vertex
 * identifiers (compute personalized PageRank).
 * @param [in]  personalization_values Pointer to an array storing personalization values for the
 * vertices in the personalization set.
 * @param [in]  alpha       PageRank damping factor.
 * @param [in]  epsilon     Error tolerance to check convergence. Convergence is assumed
 *                          if the sum of the differences in PageRank values between two
 *                          consecutive iterations is less than the number of vertices
 *                          in the graph multiplied by @p epsilon.
 * @param [in]  max_iterations Maximum number of PageRank iterations.
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result      Opaque pointer to pagerank results
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status rocgraph_personalized_pagerank_allow_nonconvergence(
    const rocgraph_handle_t*                        handle,
    rocgraph_graph_t*                               graph,
    const rocgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices,
    const rocgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
    const rocgraph_type_erased_device_array_view_t* initial_guess_vertices,
    const rocgraph_type_erased_device_array_view_t* initial_guess_values,
    const rocgraph_type_erased_device_array_view_t* personalization_vertices,
    const rocgraph_type_erased_device_array_view_t* personalization_values,
    double                                          alpha,
    double                                          epsilon,
    size_t                                          max_iterations,
    rocgraph_bool                                   do_expensive_check,
    rocgraph_centrality_result_t**                  result,
    rocgraph_error_t**                              error);

/**
 * @brief     Compute eigenvector centrality
 *
 * Computed using the power method.
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  graph       Pointer to graph
 * @param [in]  epsilon     Error tolerance to check convergence. Convergence is measured
 *                          comparing the L1 norm until it is less than epsilon
 * @param [in]  max_iterations Maximum number of power iterations, will not exceed this number
 *                          of iterations even if we haven't converged
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result      Opaque pointer to eigenvector centrality results
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_eigenvector_centrality(const rocgraph_handle_t*       handle,
                                    rocgraph_graph_t*              graph,
                                    double                         epsilon,
                                    size_t                         max_iterations,
                                    rocgraph_bool                  do_expensive_check,
                                    rocgraph_centrality_result_t** result,
                                    rocgraph_error_t**             error);

/**
 * @brief     Compute katz centrality
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  graph       Pointer to graph
 * @param [in]  betas       Optionally send in a device array holding values to be added to
 *                          each vertex's new Katz Centrality score in every iteration.
 *                          If set to NULL then @p beta is used for all vertices.
 * @param [in]  alpha       Katz centrality attenuation factor.  This should be smaller
 *                          than the inverse of the maximum eigenvalue of this graph
 * @param [in]  beta        Constant value to be added to each vertex's new Katz
 *                          Centrality score in every iteration.  Relevant only when
 *                          @p betas is NULL
 * @param [in]  epsilon     Error tolerance to check convergence. Convergence is assumed
 *                          if the sum of the differences in Katz Centrality values between
 *                          two consecutive iterations is less than the number of vertices
 *                          in the graph multiplied by @p epsilon. (L1-norm)
 * @param [in]  max_iterations Maximum number of Katz Centrality iterations.
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result      Opaque pointer to katz centrality results
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_katz_centrality(const rocgraph_handle_t*                        handle,
                             rocgraph_graph_t*                               graph,
                             const rocgraph_type_erased_device_array_view_t* betas,
                             double                                          alpha,
                             double                                          beta,
                             double                                          epsilon,
                             size_t                                          max_iterations,
                             rocgraph_bool                                   do_expensive_check,
                             rocgraph_centrality_result_t**                  result,
                             rocgraph_error_t**                              error);

/**
 * @brief     Compute betweenness centrality
 *
 * Betweenness can be computed exactly by specifying vertex_list as NULL.  This will compute
 * betweenness centrality by doing a traversal from every source vertex.
 *
 * Approximate betweenness can be computed specifying a list of vertices that should be
 * used as seeds for the traversals.  Note that the function rocgraph_select_random_vertices can be
 * used to create a list of seeds.
 *
 * @param [in]  handle             Handle for accessing resources
 * @param [in]  graph              Pointer to graph
 * @param [in]  vertex_list        Optionally specify a device array containing a list of vertices
 *                                 to use as seeds for betweenness centrality approximation
 * @param [in]  normalized         Normalize
 * @param [in]  include_endpoints  The traditional formulation of betweenness centrality does not
 *                                 include endpoints when considering a vertex to be on a shortest
 *                                 path.  Setting this to true will consider the endpoints of a
 *                                 path to be part of the path.
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result             Opaque pointer to betweenness centrality results
 * @param [out] error              Pointer to an error object storing details of any error.  Will
 *                                 be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_betweenness_centrality(const rocgraph_handle_t*                        handle,
                                    rocgraph_graph_t*                               graph,
                                    const rocgraph_type_erased_device_array_view_t* vertex_list,
                                    rocgraph_bool                                   normalized,
                                    rocgraph_bool                  include_endpoints,
                                    rocgraph_bool                  do_expensive_check,
                                    rocgraph_centrality_result_t** result,
                                    rocgraph_error_t**             error);

/**
 * @brief     Compute edge betweenness centrality
 *
 * Edge betweenness can be computed exactly by specifying vertex_list as NULL.  This will compute
 * betweenness centrality by doing a traversal from every vertex and counting the frequency that a
 * edge appears on a shortest path.
 *
 * Approximate betweenness can be computed specifying a list of vertices that should be
 * used as seeds for the traversals.  Note that the function rocgraph_select_random_vertices can be
 * used to create a list of seeds.
 *
 * @param [in]  handle             Handle for accessing resources
 * @param [in]  graph              Pointer to graph
 * @param [in]  vertex_list        Optionally specify a device array containing a list of vertices
 *                                 to use as seeds for betweenness centrality approximation
 * @param [in]  normalized         Normalize
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result             Opaque pointer to edge betweenness centrality results
 * @param [out] error              Pointer to an error object storing details of any error.  Will
 *                                 be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status rocgraph_edge_betweenness_centrality(
    const rocgraph_handle_t*                        handle,
    rocgraph_graph_t*                               graph,
    const rocgraph_type_erased_device_array_view_t* vertex_list,
    rocgraph_bool                                   normalized,
    rocgraph_bool                                   do_expensive_check,
    rocgraph_edge_centrality_result_t**             result,
    rocgraph_error_t**                              error);

/**
 * @brief     Compute hits
 *
 * @param [in]  handle      Handle for accessing resources.
 * @param [in]  graph       Pointer to graph.
 * @param [in]  epsilon     Error tolerance to check convergence. Convergence is assumed
 *                          if the sum of the differences in Hits values between two
 *                          consecutive iterations is less than the number of vertices
 *                          in the graph multiplied by @p epsilon.
 * @param [in]  max_iterations
 *                          Maximum number of Hits iterations.
 * @param [in]  initial_hubs_guess_vertices
 *                          Pointer to optional type erased device array containing
 *                          the vertex ids for an initial hubs guess.  If set to NULL
 *                          there is no initial guess.
 * @param [in]  initial_hubs_guess_values
 *                          Pointer to optional type erased device array containing
 *                          the values for an initial hubs guess.  If set to NULL
 *                          there is no initial guess.  Note that both
 *                          @p initial_hubs_guess_vertices and @p initial_hubs_guess_values
 *                          have to be specified (or they both have to be NULL).  Otherwise
 *                          this will be treated as an error.
 * @param [in]  normalize   A flag to normalize the results (if set to `true`)
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result      Opaque pointer to hits results.
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not ROCGRAPH_SUCCESS
 * @return error            Returned error code.
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_hits(const rocgraph_handle_t*                        handle,
                  rocgraph_graph_t*                               graph,
                  double                                          epsilon,
                  size_t                                          max_iterations,
                  const rocgraph_type_erased_device_array_view_t* initial_hubs_guess_vertices,
                  const rocgraph_type_erased_device_array_view_t* initial_hubs_guess_values,
                  rocgraph_bool                                   normalize,
                  rocgraph_bool                                   do_expensive_check,
                  rocgraph_hits_result_t**                        result,
                  rocgraph_error_t**                              error);

#ifdef __cplusplus
}
#endif
