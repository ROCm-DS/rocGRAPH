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

#include "internal/types/rocgraph_hierarchical_clustering_result_t.h"
#include "internal/types/rocgraph_type_erased_device_array_view_t.h"
#include "rocgraph-export.h"

/** @ingroup aux_module
 */
#ifdef __cplusplus
extern "C" {
#endif

/**
   * @ingroup community
   * @brief     Get hierarchical clustering vertices
   */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_hierarchical_clustering_result_get_vertices(
        rocgraph_hierarchical_clustering_result_t* result);

/**
   * @ingroup community
   * @brief     Get hierarchical clustering clusters
   */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_hierarchical_clustering_result_get_clusters(
        rocgraph_hierarchical_clustering_result_t* result);

/**
   * @ingroup community
   * @brief     Get modularity
   */
ROCGRAPH_EXPORT double rocgraph_hierarchical_clustering_result_get_modularity(
    rocgraph_hierarchical_clustering_result_t* result);

/**
   * @ingroup community
   * @brief     Free a hierarchical clustering result
   *
   * @param [in] result     The result from a sampling algorithm
   */
ROCGRAPH_EXPORT void
    rocgraph_hierarchical_clustering_result_free(rocgraph_hierarchical_clustering_result_t* result);

#ifdef __cplusplus
}
#endif
