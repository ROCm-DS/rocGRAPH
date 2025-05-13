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
#include "internal/types/rocgraph_degrees_result_t.h"
#include "internal/types/rocgraph_error_t.h"
#include "internal/types/rocgraph_graph_t.h"
#include "internal/types/rocgraph_handle_t.h"
#include "internal/types/rocgraph_induced_subgraph_result_t.h"
#include "internal/types/rocgraph_status.h"
#include "internal/types/rocgraph_vertex_pairs_t.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief       Get the first vertex id array
 *
 * @param [in]     vertex_pairs   A vertex_pairs
 * @return type erased array of vertex ids
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_vertex_pairs_get_first(rocgraph_vertex_pairs_t* vertex_pairs);

/**
 * @brief       Get the second vertex id array
 *
 * @param [in]     vertex_pairs   A vertex_pairs
 * @return type erased array of vertex ids
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_vertex_pairs_get_second(rocgraph_vertex_pairs_t* vertex_pairs);

/**
 * @brief     Free vertex pair
 *
 * @param [in]    vertex_pairs The vertex pairs
 */
ROCGRAPH_EXPORT void rocgraph_vertex_pairs_free(rocgraph_vertex_pairs_t* vertex_pairs);

#ifdef __cplusplus
}
#endif
