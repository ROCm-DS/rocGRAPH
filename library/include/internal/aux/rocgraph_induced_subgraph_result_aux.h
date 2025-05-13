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

#include "internal/types/rocgraph_induced_subgraph_result_t.h"
#include "internal/types/rocgraph_type_erased_device_array_view_t.h"
#include "rocgraph-export.h"

/** @ingroup aux_module
 */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief       Get the source vertex ids
 *
 * @param [in]     induced_subgraph   Opaque pointer to induced subgraph
 * @return type erased array view of source vertex ids
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_induced_subgraph_get_sources(rocgraph_induced_subgraph_result_t* induced_subgraph);

/**
 * @brief       Get the destination vertex ids
 *
 * @param [in]     induced_subgraph   Opaque pointer to induced subgraph
 * @return type erased array view of destination vertex ids
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_induced_subgraph_get_destinations(
        rocgraph_induced_subgraph_result_t* induced_subgraph);

/**
 * @brief       Get the edge weights
 *
 * @param [in]     induced_subgraph   Opaque pointer to induced subgraph
 * @return type erased array view of edge weights
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_induced_subgraph_get_edge_weights(
        rocgraph_induced_subgraph_result_t* induced_subgraph);

/**
 * @brief       Get the edge ids
 *
 * @param [in]     induced_subgraph   Opaque pointer to induced subgraph
 * @return type erased array view of edge ids
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_induced_subgraph_get_edge_ids(rocgraph_induced_subgraph_result_t* induced_subgraph);

/**
 * @brief       Get the edge types
 *
 * @param [in]     induced_subgraph   Opaque pointer to induced subgraph
 * @return type erased array view of edge types
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_induced_subgraph_get_edge_type_ids(
        rocgraph_induced_subgraph_result_t* induced_subgraph);

/**
 * @brief       Get the subgraph offsets
 *
 * @param [in]     induced_subgraph   Opaque pointer to induced subgraph
 * @return type erased array view of subgraph identifiers
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_induced_subgraph_get_subgraph_offsets(
        rocgraph_induced_subgraph_result_t* induced_subgraph);

/**
 * @brief     Free induced subgraph
 *
 * @param [in]    induced subgraph   Opaque pointer to induced subgraph
 */
ROCGRAPH_EXPORT void
    rocgraph_induced_subgraph_result_free(rocgraph_induced_subgraph_result_t* induced_subgraph);

#ifdef __cplusplus
}
#endif
