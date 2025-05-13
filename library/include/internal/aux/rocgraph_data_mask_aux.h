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

#include "internal/types/rocgraph_handle_t.h"
#include "internal/types/rocgraph_type_erased_device_array_view_t.h"
#include "rocgraph-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief     Create a data mask
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  vertex_bit_mask Device array containing vertex bit mask
 * @param [in]  edge_bit_mask   Device array containing edge bit mask
 * @param [in]  complement      If true, a 0 in one of the bit masks implies
 *                              the vertex/edge should be included and a 1 should
 *                              be excluded.  If false a 1 in one of the bit masks
 *                              implies the vertex/edge should be included and a 0
 *                              should be excluded.
 * @param [out] mask            An opaque pointer to the constructed mask object
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status
    rocgraph_data_mask_create(const rocgraph_handle_t*                        handle,
                              const rocgraph_type_erased_device_array_view_t* vertex_bit_mask,
                              const rocgraph_type_erased_device_array_view_t* edge_bit_mask,
                              rocgraph_bool                                   complement,
                              rocgraph_data_mask_t**                          mask,
                              rocgraph_error_t**                              error);

/**
 * @brief     Destroy a data mask
 *
 * @param [in]  mask  A pointer to the data mask to destroy
 */
ROCGRAPH_EXPORT void rocgraph_data_mask_destroy(rocgraph_data_mask_t* mask);

#ifdef __cplusplus
}
#endif
