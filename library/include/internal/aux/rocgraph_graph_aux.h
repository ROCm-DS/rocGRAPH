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

#include "rocgraph-export.h"

#include "internal/types/rocgraph_data_mask_t.h"
#include "internal/types/rocgraph_error_t.h"
#include "internal/types/rocgraph_graph_properties_t.h"
#include "internal/types/rocgraph_graph_t.h"
#include "internal/types/rocgraph_handle_t.h"
#include "internal/types/rocgraph_status.h"
#include "internal/types/rocgraph_type_erased_device_array_view_t.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief     Construct an SG graph
 *
 * @deprecated  This API will be deleted, use rocgraph_graph_create_sg instead
 *
 * @param [in]  handle         Handle for accessing resources
 * @param [in]  properties     Properties of the constructed graph
 * @param [in]  src            Device array containing the source vertex ids.
 * @param [in]  dst            Device array containing the destination vertex ids
 * @param [in]  weights        Device array containing the edge weights.  Note that an unweighted
 *                             graph can be created by passing weights == NULL.
 * @param [in]  edge_ids       Device array containing the edge ids for each edge.  Optional
                               argument that can be NULL if edge ids are not used.
 * @param [in]  edge_type_ids  Device array containing the edge types for each edge.  Optional
                               argument that can be NULL if edge types are not used.
 * @param [in]  store_transposed If true create the graph initially in transposed format
 * @param [in]  renumber       If true, renumber vertices to make an efficient data structure.
 *    If false, do not renumber.  Renumbering enables some significant optimizations within
 *    the graph primitives library, so it is strongly encouraged.  Renumbering is required if
 *    the vertices are not sequential integer values from 0 to num_vertices.
 * @param [in]  do_expensive_check    If true, do expensive checks to validate the input data
 *    is consistent with software assumptions.  If false bypass these checks.
 * @param [out] graph          A pointer to the graph object
 * @param [out] error          Pointer to an error object storing details of any error.  Will
 *                             be populated if error code is not ROCGRAPH_SUCCESS
 *
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_sg_graph_create(const rocgraph_handle_t*                        handle,
                             const rocgraph_graph_properties_t*              properties,
                             const rocgraph_type_erased_device_array_view_t* src,
                             const rocgraph_type_erased_device_array_view_t* dst,
                             const rocgraph_type_erased_device_array_view_t* weights,
                             const rocgraph_type_erased_device_array_view_t* edge_ids,
                             const rocgraph_type_erased_device_array_view_t* edge_type_ids,
                             rocgraph_bool                                   store_transposed,
                             rocgraph_bool                                   renumber,
                             rocgraph_bool                                   do_expensive_check,
                             rocgraph_graph_t**                              graph,
                             rocgraph_error_t**                              error);

/**
 * @brief     Construct an SG graph
 *
 * @param [in]  handle         Handle for accessing resources
 * @param [in]  properties     Properties of the constructed graph
 * @param [in]  vertices       Optional device array containing a list of vertex ids
 *                             (specify NULL if we should create vertex ids from the
 *                             unique contents of @p src and @p dst)
 * @param [in]  src            Device array containing the source vertex ids.
 * @param [in]  dst            Device array containing the destination vertex ids
 * @param [in]  weights        Device array containing the edge weights.  Note that an unweighted
 *                             graph can be created by passing weights == NULL.
 * @param [in]  edge_ids       Device array containing the edge ids for each edge.  Optional
                               argument that can be NULL if edge ids are not used.
 * @param [in]  edge_type_ids  Device array containing the edge types for each edge.  Optional
                               argument that can be NULL if edge types are not used.
 * @param [in]  store_transposed If true create the graph initially in transposed format
 * @param [in]  renumber       If true, renumber vertices to make an efficient data structure.
 *    If false, do not renumber.  Renumbering enables some significant optimizations within
 *    the graph primitives library, so it is strongly encouraged.  Renumbering is required if
 *    the vertices are not sequential integer values from 0 to num_vertices.
 * @param [in]  drop_self_loops  If true, drop any self loops that exist in the provided edge list.
 * @param [in]  drop_multi_edges If true, drop any multi edges that exist in the provided edge list.
 *    Note that setting this flag will arbitrarily select one instance of a multi edge to be the
 *    edge that survives.  If the edges have properties that should be honored (e.g. sum the
 weights,
 *    or take the maximum weight), the caller should remove specific edges themselves and not rely
 *    on this flag.
 * @param [in]  do_expensive_check    If true, do expensive checks to validate the input data
 *    is consistent with software assumptions.  If false bypass these checks.
 * @param [out] graph          A pointer to the graph object
 * @param [out] error          Pointer to an error object storing details of any error.  Will
 *                             be populated if error code is not ROCGRAPH_SUCCESS
 *
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_graph_create_sg(const rocgraph_handle_t*                        handle,
                             const rocgraph_graph_properties_t*              properties,
                             const rocgraph_type_erased_device_array_view_t* vertices,
                             const rocgraph_type_erased_device_array_view_t* src,
                             const rocgraph_type_erased_device_array_view_t* dst,
                             const rocgraph_type_erased_device_array_view_t* weights,
                             const rocgraph_type_erased_device_array_view_t* edge_ids,
                             const rocgraph_type_erased_device_array_view_t* edge_type_ids,
                             rocgraph_bool                                   store_transposed,
                             rocgraph_bool                                   renumber,
                             rocgraph_bool                                   drop_self_loops,
                             rocgraph_bool                                   drop_multi_edges,
                             rocgraph_bool                                   do_expensive_check,
                             rocgraph_graph_t**                              graph,
                             rocgraph_error_t**                              error);

/**
 * @brief     Construct an SG graph from a CSR input
 *
 * @deprecated  This API will be deleted, use rocgraph_graph_create_sg_from_csr instead
 *
 * @param [in]  handle         Handle for accessing resources
 * @param [in]  properties     Properties of the constructed graph
 * @param [in]  offsets        Device array containing the CSR offsets array
 * @param [in]  indices        Device array containing the destination vertex ids
 * @param [in]  weights        Device array containing the edge weights.  Note that an unweighted
 *                             graph can be created by passing weights == NULL.
 * @param [in]  edge_ids       Device array containing the edge ids for each edge.  Optional
                               argument that can be NULL if edge ids are not used.
 * @param [in]  edge_type_ids  Device array containing the edge types for each edge.  Optional
                               argument that can be NULL if edge types are not used.
 * @param [in]  store_transposed If true create the graph initially in transposed format
 * @param [in]  renumber       If true, renumber vertices to make an efficient data structure.
 *    If false, do not renumber.  Renumbering enables some significant optimizations within
 *    the graph primitives library, so it is strongly encouraged.  Renumbering is required if
 *    the vertices are not sequential integer values from 0 to num_vertices.
 * @param [in]  do_expensive_check    If true, do expensive checks to validate the input data
 *    is consistent with software assumptions.  If false bypass these checks.
 * @param [out] graph          A pointer to the graph object
 * @param [out] error          Pointer to an error object storing details of any error.  Will
 *                             be populated if error code is not ROCGRAPH_SUCCESS
 *
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_sg_graph_create_from_csr(const rocgraph_handle_t*                        handle,
                                      const rocgraph_graph_properties_t*              properties,
                                      const rocgraph_type_erased_device_array_view_t* offsets,
                                      const rocgraph_type_erased_device_array_view_t* indices,
                                      const rocgraph_type_erased_device_array_view_t* weights,
                                      const rocgraph_type_erased_device_array_view_t* edge_ids,
                                      const rocgraph_type_erased_device_array_view_t* edge_type_ids,
                                      rocgraph_bool      store_transposed,
                                      rocgraph_bool      renumber,
                                      rocgraph_bool      do_expensive_check,
                                      rocgraph_graph_t** graph,
                                      rocgraph_error_t** error);

/**
 * @brief     Construct an SG graph from a CSR input
 *
 * @param [in]  handle         Handle for accessing resources
 * @param [in]  properties     Properties of the constructed graph
 * @param [in]  offsets        Device array containing the CSR offsets array
 * @param [in]  indices        Device array containing the destination vertex ids
 * @param [in]  weights        Device array containing the edge weights.  Note that an unweighted
 *                             graph can be created by passing weights == NULL.
 * @param [in]  edge_ids       Device array containing the edge ids for each edge.  Optional
                               argument that can be NULL if edge ids are not used.
 * @param [in]  edge_type_ids  Device array containing the edge types for each edge.  Optional
                               argument that can be NULL if edge types are not used.
 * @param [in]  store_transposed If true create the graph initially in transposed format
 * @param [in]  renumber       If true, renumber vertices to make an efficient data structure.
 *    If false, do not renumber.  Renumbering enables some significant optimizations within
 *    the graph primitives library, so it is strongly encouraged.  Renumbering is required if
 *    the vertices are not sequential integer values from 0 to num_vertices.
 * @param [in]  do_expensive_check    If true, do expensive checks to validate the input data
 *    is consistent with software assumptions.  If false bypass these checks.
 * @param [out] graph          A pointer to the graph object
 * @param [out] error          Pointer to an error object storing details of any error.  Will
 *                             be populated if error code is not ROCGRAPH_SUCCESS
 *
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_graph_create_sg_from_csr(const rocgraph_handle_t*                        handle,
                                      const rocgraph_graph_properties_t*              properties,
                                      const rocgraph_type_erased_device_array_view_t* offsets,
                                      const rocgraph_type_erased_device_array_view_t* indices,
                                      const rocgraph_type_erased_device_array_view_t* weights,
                                      const rocgraph_type_erased_device_array_view_t* edge_ids,
                                      const rocgraph_type_erased_device_array_view_t* edge_type_ids,
                                      rocgraph_bool      store_transposed,
                                      rocgraph_bool      renumber,
                                      rocgraph_bool      do_expensive_check,
                                      rocgraph_graph_t** graph,
                                      rocgraph_error_t** error);

/**
 * @brief     Construct an MG graph
 *
 * @deprecated  This API will be deleted, use rocgraph_graph_create_mg instead
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  properties      Properties of the constructed graph
 * @param [in]  src             Device array containing the source vertex ids
 * @param [in]  dst             Device array containing the destination vertex ids
 * @param [in]  weights         Device array containing the edge weights.  Note that an unweighted
 *                              graph can be created by passing weights == NULL.  If a weighted
 *                              graph is to be created, the weights device array should be created
 *                              on each rank, but the pointer can be NULL and the size 0
 *                              if there are no inputs provided by this rank
 * @param [in]  edge_ids        Device array containing the edge ids for each edge.  Optional
                                argument that can be NULL if edge ids are not used.
 * @param [in]  edge_type_ids  Device array containing the edge types for each edge.  Optional
                                argument that can be NULL if edge types are not used.
 * @param [in]  store_transposed If true create the graph initially in transposed format
 * @param [in]  num_edges       Number of edges
 * @param [in]  do_expensive_check  If true, do expensive checks to validate the input data
 *    is consistent with software assumptions.  If false bypass these checks.
 * @param [out] graph           A pointer to the graph object
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_mg_graph_create(const rocgraph_handle_t*                        handle,
                             const rocgraph_graph_properties_t*              properties,
                             const rocgraph_type_erased_device_array_view_t* src,
                             const rocgraph_type_erased_device_array_view_t* dst,
                             const rocgraph_type_erased_device_array_view_t* weights,
                             const rocgraph_type_erased_device_array_view_t* edge_ids,
                             const rocgraph_type_erased_device_array_view_t* edge_type_ids,
                             rocgraph_bool                                   store_transposed,
                             size_t                                          num_edges,
                             rocgraph_bool                                   do_expensive_check,
                             rocgraph_graph_t**                              graph,
                             rocgraph_error_t**                              error);

/**
 * @brief     Construct an MG graph
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  properties      Properties of the constructed graph
 * @param [in]  vertices        List of device arrays containing the unique vertex ids.
 *                              If NULL we will construct this internally using the unique
 *                              entries specified in src and dst
 *                              All entries in this list will be concatenated on this GPU
 *                              into a single array.
 * @param [in]  src             List of device array containing the source vertex ids
 *                              All entries in this list will be concatenated on this GPU
 *                              into a single array.
 * @param [in]  dst             List of device array containing the destination vertex ids
 *                              All entries in this list will be concatenated on this GPU
 *                              into a single array.
 * @param [in]  weights         List of device array containing the edge weights.  Note that an
 * unweighted graph can be created by passing weights == NULL.  If a weighted graph is to be
 * created, the weights device array should be created on each rank, but the pointer can be NULL and
 * the size 0 if there are no inputs provided by this rank All entries in this list will be
 * concatenated on this GPU into a single array.
 * @param [in]  edge_ids        List of device array containing the edge ids for each edge. Optional
 *                              argument that can be NULL if edge ids are not used.
 *                              All entries in this list will be concatenated on this GPU
 *                              into a single array.
 * @param [in]  edge_type_ids   List of device array containing the edge types for each edge.
 * Optional argument that can be NULL if edge types are not used. All entries in this list will be
 * concatenated on this GPU into a single array.
 * @param [in]  store_transposed If true create the graph initially in transposed format
 * @param [in]  num_arrays      The number of arrays specified in @p vertices, @p src, @p dst, @p
 *                              weights, @p edge_ids and @p edge_type_ids
 * @param [in]  drop_self_loops  If true, drop any self loops that exist in the provided edge list.
 * @param [in]  drop_multi_edges If true, drop any multi edges that exist in the provided edge list.
 *    Note that setting this flag will arbitrarily select one instance of a multi edge to be the
 *    edge that survives.  If the edges have properties that should be honored (e.g. sum the
 * weights, or take the maximum weight), the caller should do that on not rely on this flag.
 * @param [in]  do_expensive_check  If true, do expensive checks to validate the input data
 *    is consistent with software assumptions.  If false bypass these checks.
 * @param [out] graph           A pointer to the graph object
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_graph_create_mg(rocgraph_handle_t const*                               handle,
                             rocgraph_graph_properties_t const*                     properties,
                             rocgraph_type_erased_device_array_view_t const* const* vertices,
                             rocgraph_type_erased_device_array_view_t const* const* src,
                             rocgraph_type_erased_device_array_view_t const* const* dst,
                             rocgraph_type_erased_device_array_view_t const* const* weights,
                             rocgraph_type_erased_device_array_view_t const* const* edge_ids,
                             rocgraph_type_erased_device_array_view_t const* const* edge_type_ids,
                             rocgraph_bool      store_transposed,
                             size_t             num_arrays,
                             rocgraph_bool      drop_self_loops,
                             rocgraph_bool      drop_multi_edges,
                             rocgraph_bool      do_expensive_check,
                             rocgraph_graph_t** graph,
                             rocgraph_error_t** error);

/**
 * @brief     Destroy an graph
 *
 * @param [in]  graph  A pointer to the graph object to destroy
 */
ROCGRAPH_EXPORT void rocgraph_graph_free(rocgraph_graph_t* graph);

/**
 * @brief     Destroy an SG graph
 *
 * @deprecated  This API will be deleted, use rocgraph_graph_free instead
 *
 * @param [in]  graph  A pointer to the graph object to destroy
 */
ROCGRAPH_EXPORT void rocgraph_sg_graph_free(rocgraph_graph_t* graph);

/**
 * @brief     Destroy an MG graph
 *
 * @deprecated  This API will be deleted, use rocgraph_graph_free instead
 *
 * @param [in]  graph  A pointer to the graph object to destroy
 */
ROCGRAPH_EXPORT void rocgraph_mg_graph_free(rocgraph_graph_t* graph);

/**
 * @brief     Get the data mask currently associated with a graph
 *
 * @param [in]  graph       The input graph
 * @param [out] mask        Opaque pointer where we should store the
 *                          current mask.  Will be NULL if there is no mask
 *                          currently assigned to the graph.
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status rocgraph_graph_get_data_mask(rocgraph_graph_t*      graph,
                                                             rocgraph_data_mask_t** mask,
                                                             rocgraph_error_t**     error);

/**
 * @brief     Associate a data mask with a graph
 *
 * NOTE: This function will fail if there is already a data mask associated with this graph
 *
 * @param [in]  graph       The input graph
 * @param [out] mask        Opaque pointer of the new data mask
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status rocgraph_graph_add_data_mask(rocgraph_graph_t*     graph,
                                                             rocgraph_data_mask_t* mask,
                                                             rocgraph_error_t**    error);

/**
 * @brief     Release the data mask currently associated with a graph
 *
 * This function will remove the associated of the current data mask
 * with this graph.  The caller will be responsible for destroying the data
 * mask using graph_data_mask_destroy.
 *
 * If this function is not called and the graph is destroyed, the act of destroying
 * the graph will also destroy the data mask.
 *
 * If this function is called on a graph that is not currently associated with
 * a graph, then the mask will be set to NULL.
 *
 * @param [in]  graph       The input graph
 * @param [out] mask        Opaque pointer where we should store the
 *                          current mask.
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status rocgraph_graph_release_data_mask(rocgraph_graph_t*      graph,
                                                                 rocgraph_data_mask_t** mask,
                                                                 rocgraph_error_t**     error);

#ifdef __cplusplus
}
#endif
