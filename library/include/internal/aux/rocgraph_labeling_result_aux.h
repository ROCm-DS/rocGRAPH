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

#include "internal/types/rocgraph_labeling_result_t.h"
#include "internal/types/rocgraph_type_erased_device_array_view_t.h"
#include "rocgraph-export.h"

/** @ingroup aux_module
 */
#ifdef __cplusplus
extern "C" {
#endif

/**
   * @ingroup labeling
   * @brief     Get the vertex ids from the labeling result
 *
 * @param [in]   result   The result from a labeling algorithm
 * @return type erased array of vertex ids
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_labeling_result_get_vertices(rocgraph_labeling_result_t* result);

/**
 * @ingroup labeling
 * @brief     Get the label values from the labeling result
 *
 * @param [in]   result   The result from a labeling algorithm
 * @return type erased array of label values
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_labeling_result_get_labels(rocgraph_labeling_result_t* result);

/**
 * @ingroup labeling
 * @brief     Free labeling result
 *
 * @param [in]   result   The result from a labeling algorithm
 */
ROCGRAPH_EXPORT void rocgraph_labeling_result_free(rocgraph_labeling_result_t* result);

#ifdef __cplusplus
}
#endif
