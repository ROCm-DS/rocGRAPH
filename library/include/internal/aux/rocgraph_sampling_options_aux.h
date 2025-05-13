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

#include "internal/types/rocgraph_bool.h"
#include "internal/types/rocgraph_compression_type.h"
#include "internal/types/rocgraph_prior_sources_behavior.h"
#include "internal/types/rocgraph_sample_result_t.h"
#include "internal/types/rocgraph_sampling_options_t.h"
#include "rocgraph-export.h"

/** @ingroup aux_module
 */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup samplingC
 * @brief   Create sampling options object
 *
 * All sampling options set to FALSE
 *
 * @param [out] options Opaque pointer to the sampling options
 * @param [out] error   Pointer to an error object storing details of any error.  Will
 *                      be populated if error code is not ROCGRAPH_SUCCESS
 */
ROCGRAPH_EXPORT rocgraph_status rocgraph_sampling_options_create(
    rocgraph_sampling_options_t** options, rocgraph_error_t** error);

/**
 * @ingroup samplingC
 * @brief   Set flag to retain seeds (original sources)
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Boolean value to assign to the option
 */
ROCGRAPH_EXPORT void rocgraph_sampling_set_retain_seeds(rocgraph_sampling_options_t* options,
                                                        rocgraph_bool                value);

/**
 * @ingroup samplingC
 * @brief   Set flag to renumber results
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Boolean value to assign to the option
 */
ROCGRAPH_EXPORT void rocgraph_sampling_set_renumber_results(rocgraph_sampling_options_t* options,
                                                            rocgraph_bool                value);

/**
 * @ingroup samplingC
 * @brief   Set whether to compress per-hop (True) or globally (False)
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Boolean value to assign to the option
 */
ROCGRAPH_EXPORT void rocgraph_sampling_set_compress_per_hop(rocgraph_sampling_options_t* options,
                                                            rocgraph_bool                value);

/**
 * @ingroup samplingC
 * @brief   Set flag to sample with_replacement
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Boolean value to assign to the option
 */
ROCGRAPH_EXPORT void rocgraph_sampling_set_with_replacement(rocgraph_sampling_options_t* options,
                                                            rocgraph_bool                value);

/**
 * @ingroup samplingC
 * @brief   Set flag to sample return_hops
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Boolean value to assign to the option
 */
ROCGRAPH_EXPORT void rocgraph_sampling_set_return_hops(rocgraph_sampling_options_t* options,
                                                       rocgraph_bool                value);

/**
 * @ingroup samplingC
 * @brief   Set compression type
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Enum defining the compresion type
 */
ROCGRAPH_EXPORT void rocgraph_sampling_set_compression_type(rocgraph_sampling_options_t* options,
                                                            rocgraph_compression_type    value);

/**
 * @ingroup samplingC
 * @brief   Set prior sources behavior
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Enum defining prior sources behavior
 */
ROCGRAPH_EXPORT void
    rocgraph_sampling_set_prior_sources_behavior(rocgraph_sampling_options_t*    options,
                                                 rocgraph_prior_sources_behavior value);

/**
 * @ingroup samplingC
 * @brief   Set flag to sample dedupe_sources prior to sampling
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Boolean value to assign to the option
 */
ROCGRAPH_EXPORT void rocgraph_sampling_set_dedupe_sources(rocgraph_sampling_options_t* options,
                                                          rocgraph_bool                value);

/**
 * @ingroup samplingC
 * @brief     Free sampling options object
 *
 * @param [in]   options   Opaque pointer to sampling object
 */
ROCGRAPH_EXPORT void rocgraph_sampling_options_free(rocgraph_sampling_options_t* options);

#ifdef __cplusplus
}
#endif
