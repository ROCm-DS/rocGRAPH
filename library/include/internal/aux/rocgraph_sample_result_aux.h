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

#include "internal/types/rocgraph_sample_result_t.h"
#include "internal/types/rocgraph_type_erased_device_array_view_t.h"
#include "rocgraph-export.h"

/** @ingroup aux_module
 */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @deprecated This call should be replaced with rocgraph_sample_result_get_majors
 * @brief     Get the source vertices from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the source vertices in device memory
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_sources(const rocgraph_sample_result_t* result);

/**
 * @deprecated This call should be replaced with rocgraph_sample_result_get_minors
 * @brief     Get the destination vertices from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the destination vertices in device memory
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_destinations(const rocgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the major vertices from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the major vertices in device memory
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_majors(const rocgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the minor vertices from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the minor vertices in device memory
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_minors(const rocgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the major offsets from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the major offsets in device memory
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_major_offsets(const rocgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the start labels from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the start labels
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_start_labels(const rocgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the edge_id from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the edge_id
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_edge_id(const rocgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the edge_type from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the edge_type
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_edge_type(const rocgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the edge_weight from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the edge_weight
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_edge_weight(const rocgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the hop from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the hop
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_hop(const rocgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the label-hop offsets from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the label-hop offsets
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_label_hop_offsets(const rocgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the index from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the index
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_index(const rocgraph_sample_result_t* result);

/**
 * @deprecated This call should be replaced with rocgraph_sample_get_get_label_hop_offsets
 * @brief     Get the result offsets from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the result offsets
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_offsets(const rocgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the renumber map
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the renumber map
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_renumber_map(const rocgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the renumber map offsets
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the renumber map offsets
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_sample_result_get_renumber_map_offsets(const rocgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Free a sampling result
 *
 * @param [in]   result   The result from a sampling algorithm
 */
ROCGRAPH_EXPORT void rocgraph_sample_result_free(rocgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Create a sampling result (testing API)
 *
 * @param [in]   handle         Handle for accessing resources
 * @param [in]   srcs           Device array view to populate srcs
 * @param [in]   dsts           Device array view to populate dsts
 * @param [in]   edge_id        Device array view to populate edge_id (can be NULL)
 * @param [in]   edge_type      Device array view to populate edge_type (can be NULL)
 * @param [in]   wgt            Device array view to populate wgt (can be NULL)
 * @param [in]   hop            Device array view to populate hop
 * @param [in]   label          Device array view to populate label (can be NULL)
 * @param [out]  result         Pointer to the location to store the
 *                              rocgraph_sample_result_t*
 * @param [out]  error          Pointer to an error object storing details of
 *                              any error.  Will be populated if error code is
 *                              not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_test_sample_result_create(const rocgraph_handle_t*                        handle,
                                       const rocgraph_type_erased_device_array_view_t* srcs,
                                       const rocgraph_type_erased_device_array_view_t* dsts,
                                       const rocgraph_type_erased_device_array_view_t* edge_id,
                                       const rocgraph_type_erased_device_array_view_t* edge_type,
                                       const rocgraph_type_erased_device_array_view_t* wgt,
                                       const rocgraph_type_erased_device_array_view_t* hop,
                                       const rocgraph_type_erased_device_array_view_t* label,
                                       rocgraph_sample_result_t**                      result,
                                       rocgraph_error_t**                              error);

/**
 * @ingroup samplingC
 * @brief     Create a sampling result (testing API)
 *
 * @param [in]   handle         Handle for accessing resources
 * @param [in]   srcs           Device array view to populate srcs
 * @param [in]   dsts           Device array view to populate dsts
 * @param [in]   edge_id        Device array view to populate edge_id
 * @param [in]   edge_type      Device array view to populate edge_type
 * @param [in]   weight         Device array view to populate weight
 * @param [in]   hop            Device array view to populate hop
 * @param [in]   label          Device array view to populate label
 * @param [out]  result         Pointer to the location to store the
 *                              rocgraph_sample_result_t*
 * @param [out]  error          Pointer to an error object storing details of
 *                              any error.  Will be populated if error code is
 *                              not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status rocgraph_test_uniform_neighborhood_sample_result_create(
    const rocgraph_handle_t*                        handle,
    const rocgraph_type_erased_device_array_view_t* srcs,
    const rocgraph_type_erased_device_array_view_t* dsts,
    const rocgraph_type_erased_device_array_view_t* edge_id,
    const rocgraph_type_erased_device_array_view_t* edge_type,
    const rocgraph_type_erased_device_array_view_t* weight,
    const rocgraph_type_erased_device_array_view_t* hop,
    const rocgraph_type_erased_device_array_view_t* label,
    rocgraph_sample_result_t**                      result,
    rocgraph_error_t**                              error);

#ifdef __cplusplus
}
#endif
