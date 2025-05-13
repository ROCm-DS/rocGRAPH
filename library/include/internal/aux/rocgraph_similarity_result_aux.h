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

#include "internal/types/rocgraph_similarity_result_t.h"
#include "internal/types/rocgraph_type_erased_device_array_view_t.h"
#include "internal/types/rocgraph_vertex_pairs_t.h"
#include "rocgraph-export.h"

/** @defgroup similarity Similarity algorithms
 */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup similarity
 * @brief       Get vertex pair from the similarity result.
 *
 * @param [in]     result   The result from a similarity algorithm
 * @return vertex pairs
 */
ROCGRAPH_EXPORT rocgraph_vertex_pairs_t*
    rocgraph_similarity_result_get_vertex_pairs(rocgraph_similarity_result_t* result);

/**
 * @ingroup similarity
 * @brief       Get the similarity coefficient array
 *
 * @param [in]     result   The result from a similarity algorithm
 * @return type erased array of similarity coefficients
 */
ROCGRAPH_EXPORT rocgraph_type_erased_device_array_view_t*
    rocgraph_similarity_result_get_similarity(rocgraph_similarity_result_t* result);

/**
 * @ingroup similarity
 * @brief     Free similarity result
 *
 * @param [in]    result    The result from a similarity algorithm
 */
ROCGRAPH_EXPORT void rocgraph_similarity_result_free(rocgraph_similarity_result_t* result);

#ifdef __cplusplus
}
#endif
