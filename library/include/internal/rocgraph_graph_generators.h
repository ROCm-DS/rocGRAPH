// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*!\file*/
/*
 * Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 */
/*
 * Copyright (C) 2023-2024, NVIDIA CORPORATION.
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
#include "internal/types/rocgraph_coo_list_t.h"
#include "internal/types/rocgraph_coo_t.h"
#include "internal/types/rocgraph_data_type_id.h"
#include "internal/types/rocgraph_error_t.h"
#include "internal/types/rocgraph_generator_distribution.h"
#include "internal/types/rocgraph_graph_t.h"
#include "internal/types/rocgraph_handle_t.h"
#include "internal/types/rocgraph_rng_state_t.h"
#include "internal/types/rocgraph_status.h"
#include "rocgraph-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief      Generate RMAT edge list
 *
 * Returns a COO containing edges generated from the RMAT generator.
 *
 * Vertex types will be int32 if scale < 32 and int64 if scale >= 32
 *
 * @param [in]     handle             Handle for accessing resources
 * @param [in,out] rng_state          State of the random number generator, updated with each call
 * @param [in]     scale Scale factor to set the number of vertices in the graph. Vertex IDs have
 * values in [0, V), where V = 1 << @p scale.
 * @param [in]     num_edges          Number of edges to generate.
 * @param [in]     a                  a, b, c, d (= 1.0 - (a + b + c)) in the R-mat graph generator
 * (vist https://graph500.org for additional details). a, b, c, d should be non-negative and a + b +
 * c should be no larger than 1.0.
 * @param [in]     b                  a, b, c, d (= 1.0 - (a + b + c)) in the R-mat graph generator
 * (vist https://graph500.org for additional details). a, b, c, d should be non-negative and a + b +
 * c should be no larger than 1.0.
 * @param [in]     c                  a, b, c, d (= 1.0 - (a + b + c)) in the R-mat graph generator
 * (vist https://graph500.org for additional details). a, b, c, d should be non-negative and a + b +
 * c should be no larger than 1.0.
 * @param [in]     clip_and_flip      Flag controlling whether to generate edges only in the lower
 * triangular part (including the diagonal) of the graph adjacency matrix (if set to `true`) or not
 * (if set to `false`).
 * @param [in]     scramble_vertex_ids Flag controlling whether to scramble vertex ID bits
 * (if set to `true`) or not (if set to `false`); scrambling vertex ID bits breaks correlation
 * between vertex ID values and vertex degrees.
 * @param [out]    result             Opaque pointer to generated coo
 * @param [out]    error              Pointer to an error object storing details of any error.  Will
 *                                    be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status rocgraph_generate_rmat_edgelist(const rocgraph_handle_t* handle,
                                                                rocgraph_rng_state_t*    rng_state,
                                                                size_t                   scale,
                                                                size_t                   num_edges,
                                                                double                   a,
                                                                double                   b,
                                                                double                   c,
                                                                rocgraph_bool clip_and_flip,
                                                                rocgraph_bool scramble_vertex_ids,
                                                                rocgraph_coo_t**   result,
                                                                rocgraph_error_t** error);

/**
 * @brief      Generate RMAT edge lists
 *
 * Returns a COO list containing edges generated from the RMAT generator.
 *
 * Vertex types will be int32 if scale < 32 and int64 if scale >= 32
 *
 * @param [in]     handle             Handle for accessing resources
 * @param [in,out] rng_state          State of the random number generator, updated with each call
 * @param [in]     n_edgelists Number of edge lists (graphs) to generate
 * @param [in]     min_scale Scale factor to set the minimum number of verties in the graph.
 * @param [in]     max_scale Scale factor to set the maximum number of verties in the graph.
 * @param [in]     edge_factor Average number of edges per vertex to generate.
 * @param [in]     size_distribution Distribution of the graph sizes, impacts the scale parameter of
 * the R-MAT generator
 * @param [in]     edge_distribution Edges distribution for each graph, impacts how R-MAT parameters
 * a,b,c,d, are set.
 * @param [in]     clip_and_flip      Flag controlling whether to generate edges only in the lower
 * triangular part (including the diagonal) of the graph adjacency matrix (if set to `true`) or not
 * (if set to `false`).
 * @param [in]     scramble_vertex_ids Flag controlling whether to scramble vertex ID bits
 * (if set to `true`) or not (if set to `false`); scrambling vertex ID bits breaks correlation
 * between vertex ID values and vertex degrees.
 * @param [out]    result             Opaque pointer to generated coo list
 * @param [out]    error              Pointer to an error object storing details of any error.  Will
 *                                    be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_generate_rmat_edgelists(const rocgraph_handle_t*        handle,
                                     rocgraph_rng_state_t*           rng_state,
                                     size_t                          n_edgelists,
                                     size_t                          min_scale,
                                     size_t                          max_scale,
                                     size_t                          edge_factor,
                                     rocgraph_generator_distribution size_distribution,
                                     rocgraph_generator_distribution edge_distribution,
                                     rocgraph_bool                   clip_and_flip,
                                     rocgraph_bool                   scramble_vertex_ids,
                                     rocgraph_coo_list_t**           result,
                                     rocgraph_error_t**              error);

/**
 * @brief      Generate edge weights and add to an rmat edge list
 *
 * Updates a COO to contain random edge weights
 *
 * @param [in]     handle             Handle for accessing resources
 * @param [in,out] rng_state          State of the random number generator, updated with each call
 * @param [in,out] coo                Opaque pointer to the coo, weights will be added (overwriting
 * any existing weights)
 * @param [in]     dtype              The type of weight to generate (FLOAT32 or FLOAT64), ignored
 * unless include_weights is true
 * @param [in]     minimum_weight     Minimum weight value to generate
 * @param [in]     maximum_weight     Maximum weight value to generate
 * @param [out]    error              Pointer to an error object storing details of any error.  Will
 *                                    be populated if error code is not ROCGRAPH_SUCCESS
 */
ROCGRAPH_EXPORT rocgraph_status rocgraph_generate_edge_weights(const rocgraph_handle_t* handle,
                                                               rocgraph_rng_state_t*    rng_state,
                                                               rocgraph_coo_t*          coo,
                                                               rocgraph_data_type_id    dtype,
                                                               double             minimum_weight,
                                                               double             maximum_weight,
                                                               rocgraph_error_t** error);

/**
 * @brief      Add edge ids to an COO
 *
 * Updates a COO to contain edge ids.  Edges will be numbered from 0 to n-1 where n is the number of
 * edges
 *
 * @param [in]     handle             Handle for accessing resources
 * @param [in,out] coo                Opaque pointer to the coo, weights will be added (overwriting
 * any existing weights)
 * @param [in]     multi_gpu          Flag if the COO is being created on multiple GPUs
 * @param [out]    error              Pointer to an error object storing details of any error.  Will
 *                                    be populated if error code is not ROCGRAPH_SUCCESS
 */
ROCGRAPH_EXPORT rocgraph_status rocgraph_generate_edge_ids(const rocgraph_handle_t* handle,
                                                           rocgraph_coo_t*          coo,
                                                           rocgraph_bool            multi_gpu,
                                                           rocgraph_error_t**       error);

/**
 * @brief      Generate random edge types, add them to an COO
 *
 * Updates a COO to contain edge types.  Edges types will be randomly generated.
 *
 * @param [in]     handle             Handle for accessing resources
 * @param [in,out] rng_state          State of the random number generator, updated with each call
 * @param [in,out] coo                Opaque pointer to the coo, weights will be added (overwriting
 * any existing weights)
 * @param [in]     min_edge_type      Edge types will be randomly generated between min_edge_type
 * and max_edge_type
 * @param [in]     max_edge_type      Edge types will be randomly generated between min_edge_type
 * and max_edge_type
 * @param [out]    error              Pointer to an error object storing details of any error.  Will
 *                                    be populated if error code is not ROCGRAPH_SUCCESS
 */
ROCGRAPH_EXPORT rocgraph_status rocgraph_generate_edge_types(const rocgraph_handle_t* handle,
                                                             rocgraph_rng_state_t*    rng_state,
                                                             rocgraph_coo_t*          coo,
                                                             int32_t                  min_edge_type,
                                                             int32_t                  max_edge_type,
                                                             rocgraph_error_t**       error);

#ifdef __cplusplus
}
#endif
