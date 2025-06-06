// Copyright (c) 2022, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <raft/util/detail/itertools.hpp>

/**
 * Helpers inspired by the Python itertools library
 *
 */

namespace raft::util::itertools
{

    /**
 * @brief Cartesian product of the given initializer lists.
 *
 * This helper can be used to easily define input parameters in tests/benchmarks.
 * Note that it's not optimized for use with large lists / many lists in performance-critical code!
 *
 * @tparam S    Type of the output structures.
 * @tparam Args Types of the elements of the initilizer lists, matching the types of the first
 *              fields of the structure (if the structure has more fields, some might be initialized
 *              with their default value).
 * @param lists One or more initializer lists.
 * @return std::vector<S> A vector of structures containing the cartesian product.
 */
    template <typename S, typename... Args>
    std::vector<S> product(std::initializer_list<Args>... lists)
    {
        return detail::product<S>(std::index_sequence_for<Args...>(),
                                  (std::vector<Args>(lists))...);
    }

} // namespace raft::util::itertools
