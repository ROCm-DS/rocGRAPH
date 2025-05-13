// Copyright (c) 2022-2023, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <raft/core/span.hpp>

namespace raft
{

    /**
 * @defgroup device_span one-dimensional device span type
 * @{
 */

    /**
 * @brief A span class for device pointer.
 */
    template <typename T, size_t extent = std::experimental::dynamic_extent>
    using device_span = span<T, true, extent>;

    /**
 * @}
 */
} // end namespace raft
