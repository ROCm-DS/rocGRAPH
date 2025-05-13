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

#include <stddef.h>

#include "internal/types/rocgraph_hits_result_t.h"
#include "internal/types/rocgraph_type_erased_device_array_view_t.h"
#include "rocgraph-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup centrality
 * @brief     Get the vertex ids from the hits result
 *
 * @param [in]   result   The result from hits
 * @return type erased array of vertex ids
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_hits_result_get_vertices(rocgraph_hits_result_t* result);

/**
 * @ingroup centrality
 * @brief     Get the hubs values from the hits result
 *
 * @param [in]   result   The result from hits
 * @return type erased array of hubs values
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_hits_result_get_hubs(rocgraph_hits_result_t* result);

/**
 * @ingroup centrality
 * @brief     Get the authorities values from the hits result
 *
 * @param [in]   result   The result from hits
 * @return type erased array of authorities values
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_hits_result_get_authorities(rocgraph_hits_result_t* result);

/**
 * @ingroup centrality
 * @brief   Get the score differences between the last two iterations
 *
 * @param [in]   result   The result from hits
 * @return score differences
 */
ROCGRAPH_EXPORT double
    rocgraph_hits_result_get_hub_score_differences(rocgraph_hits_result_t* result);

/**
 * @ingroup centrality
 * @brief   Get the actual number of iterations
 *
 * @param [in]   result   The result from hits
 * @return actual number of iterations
 */
ROCGRAPH_EXPORT size_t
    rocgraph_hits_result_get_number_of_iterations(rocgraph_hits_result_t* result);

/**
 * @ingroup centrality
 * @brief     Free hits result
 *
 * @param [in]   result   The result from hits
 */
ROCGRAPH_EXPORT void rocgraph_hits_result_free(rocgraph_hits_result_t* result);

#ifdef __cplusplus
}
#endif
