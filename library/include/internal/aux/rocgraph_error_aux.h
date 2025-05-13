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

#include "internal/types/rocgraph_error_t.h"
#include "rocgraph-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
   * @brief     Return an error message
   *
   * @param [in]  error       The error object from some rocgraph function call
   * @return a C-style string that provides detail for the error
   */
ROCGRAPH_EXPORT const char* rocgraph_error_message(const rocgraph_error_t* error);

/**
   * @brief    Destroy an error message
   *
   * @param [in]  error       The error object from some rocgraph function call
   */
ROCGRAPH_EXPORT void rocgraph_error_free(rocgraph_error_t* error);

#ifdef __cplusplus
}
#endif
