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
 * @brief Selects the type of compression to use for the output samples.
 */
typedef enum
{
    /** Outputs in COO format.  Default. */
    rocgraph_compression_type_coo = 0,
    /** Compresses in CSR format.  This means the row (src) column
               is compressed into a row pointer. */
    rocgraph_compression_type_csr,
    /** Compresses in CSC format.  This means the col (dst) column
               is compressed into a column pointer. */
    rocgraph_compression_type_csc,
    /** Compresses in DCSR format.  This outputs an additional index
              that avoids empty entries in the row pointer. */
    rocgraph_compression_type_dcsr,
    /** Compresses in DCSC format.  This outputs an additional index
               that avoid empty entries in the col pointer. */
    rocgraph_compression_type_dcsc
} rocgraph_compression_type;

#ifdef __cplusplus
}
#endif
