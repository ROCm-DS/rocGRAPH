// Copyright (c) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

namespace raft::linalg
{

    /**
 * @brief Enum for reduction/broadcast where an operation is to be performed along
 *        a matrix's rows or columns
 *
 */
    enum class Apply
    {
        ALONG_ROWS,
        ALONG_COLUMNS
    };

    /**
 * @brief Enum for reduction/broadcast where an operation is to be performed along
 *        a matrix's rows or columns
 *
 */
    enum class FillMode
    {
        UPPER,
        LOWER
    };

    /**
 * @brief Enum for this type indicates which operation is applied to the related input (e.g. sparse
 * matrix, or vector).
 *
 */
    enum class Operation
    {
        NON_TRANSPOSE,
        TRANSPOSE
    };

} // end namespace raft::linalg
