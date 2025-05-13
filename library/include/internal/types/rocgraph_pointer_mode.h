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

#ifdef __cplusplus
extern "C" {
#endif

/** @ingroup types_module
   *  @brief Indicates if the pointer is device pointer or host pointer.
   *
   *  @enum rocgraph_pointer_mode
   *  @details
   *  The @ref rocgraph_pointer_mode indicates whether scalar values are passed by
   *  reference on the host or device. The @ref rocgraph_pointer_mode can be changed by
   *  rocgraph_set_pointer_mode(). The currently used pointer mode can be obtained by
   *  rocgraph_get_pointer_mode().
   */
typedef enum rocgraph_pointer_mode_
{
    rocgraph_pointer_mode_host   = 0, /**< scalar pointers are in host memory. */
    rocgraph_pointer_mode_device = 1 /**< scalar pointers are in device memory. */
} rocgraph_pointer_mode;

#ifdef __cplusplus
}
#endif
