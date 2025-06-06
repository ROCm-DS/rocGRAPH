// Copyright (c) 2022-2023, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <raft/core/span.hpp>

namespace raft
{

    /**
 * @defgroup host_span one-dimensional device span type
 * @{
 */

    /**
 * @brief A span class for host pointer.
 */
    template <typename T, size_t extent = std::experimental::dynamic_extent>
    using host_span = span<T, false, extent>;

    /**
 * @}
 */

} // end namespace raft
