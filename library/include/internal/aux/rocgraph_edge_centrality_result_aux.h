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

#include "internal/types/rocgraph_edge_centrality_result_t.h"
#include "internal/types/rocgraph_type_erased_device_array_view_t.h"
#include "rocgraph-export.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @ingroup centrality
 * @brief     Get the src vertex ids from an edge centrality result
 *
 * @param [in]   result   The result from an edge centrality algorithm
 * @return type erased array of src vertex ids
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_edge_centrality_result_get_src_vertices(rocgraph_edge_centrality_result_t* result);

/**
 * @ingroup centrality
 * @brief     Get the dst vertex ids from an edge centrality result
 *
 * @param [in]   result   The result from an edge centrality algorithm
 * @return type erased array of dst vertex ids
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_edge_centrality_result_get_dst_vertices(rocgraph_edge_centrality_result_t* result);

/**
 * @ingroup centrality
 * @brief     Get the edge ids from an edge centrality result
 *
 * @param [in]   result   The result from an edge centrality algorithm
 * @return type erased array of edge ids
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_edge_centrality_result_get_edge_ids(rocgraph_edge_centrality_result_t* result);

/**
 * @ingroup centrality
 * @brief     Get the centrality values from an edge centrality algorithm result
 *
 * @param [in]   result   The result from an edge centrality algorithm
 * @return type erased array view of centrality values
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_edge_centrality_result_get_values(rocgraph_edge_centrality_result_t* result);

/**
 * @ingroup centrality
 * @brief     Free centrality result
 *
 * @param [in]   result   The result from a centrality algorithm
 */
ROCGRAPH_EXPORT void
    rocgraph_edge_centrality_result_free(rocgraph_edge_centrality_result_t* result);

#ifdef __cplusplus
}
#endif
