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

#include <stddef.h>

#include "internal/types/rocgraph_coo_list_t.h"
#include "rocgraph-export.h"

/** @ingroup aux_module
 */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief       Get the number of coo object in the list
 *
 * @param [in]     coo_list   Opaque pointer to COO list
 * @return number of elements
 */
ROCGRAPH_EXPORT size_t rocgraph_coo_list_size(const rocgraph_coo_list_t* coo_list);

/**
 * @brief       Get a COO from the list
 *
 * @param [in]     coo_list   Opaque pointer to COO list
 * @param [in]     index      Index of desired COO from list
 * @return a rocgraph_coo_t* object from the list
 */
ROCGRAPH_EXPORT rocgraph_coo_t* rocgraph_coo_list_element(rocgraph_coo_list_t* coo_list,
                                                          size_t               index);
/**
 * @brief     Free coo list
 *
 * @param [in]    coo_list Opaque pointer to list of COO objects
 */
ROCGRAPH_EXPORT void rocgraph_coo_list_free(rocgraph_coo_list_t* coo_list);

#ifdef __cplusplus
}
#endif
