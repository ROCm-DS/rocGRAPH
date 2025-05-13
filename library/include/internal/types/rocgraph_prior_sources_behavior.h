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

/** @ingroup types_module
 */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief     Enumeration for prior sources behavior
 */
typedef enum
{
    /** Construct sources for hop k from destination vertices from hop k-1 */
    rocgraph_prior_sources_behavior_default = 0,
    /** Construct sources for hop k from destination vertices from hop k-1
	and sources from hop k-1 */
    rocgraph_prior_sources_behavior_carry_over,
    /** Construct sources for hop k from destination vertices form hop k-1,
	but exclude any vertex that has already been used as a source */
    rocgraph_prior_sources_behavior_exclude
} rocgraph_prior_sources_behavior;

#ifdef __cplusplus
}
#endif
