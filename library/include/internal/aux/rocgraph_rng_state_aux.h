// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*!\file*/
/*
 * Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 */
/*
 * Copyright (C) 2023-2024, NVIDIA CORPORATION.
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

#include "internal/types/rocgraph_error_t.h"
#include "internal/types/rocgraph_handle_t.h"
#include "internal/types/rocgraph_rng_state_t.h"
#include "internal/types/rocgraph_status.h"
#include "rocgraph-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief     Create a Random Number Generator State
 * @param [in]  handle      Resource handle
 * @param [in]  seed        Initial value for seed.  In MG this should be different
 *                          on each GPU
 * @param [out] state       Pointer to the location to store the pointer to the RngState
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not ROCGRAPH_SUCCESS
 * @return error code
 */
ROCGRAPH_EXPORT rocgraph_status rocgraph_rng_state_create(const rocgraph_handle_t* handle,
                                                          uint64_t                 seed,
                                                          rocgraph_rng_state_t**   state,
                                                          rocgraph_error_t**       error);

/**
 * @brief    Destroy a Random Number Generator State
 *
 * @param [in]  p    Pointer to the Random Number Generator State
 */
ROCGRAPH_EXPORT void rocgraph_rng_state_free(rocgraph_rng_state_t* p);

#ifdef __cplusplus
}
#endif
