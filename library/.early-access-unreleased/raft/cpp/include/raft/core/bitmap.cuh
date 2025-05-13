// Copyright (c) 2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <raft/core/bitmap.hpp>
#include <raft/core/bitset.cuh>
#include <raft/core/detail/mdspan_util.cuh>
#include <raft/core/device_container_policy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>

#include <type_traits>

namespace raft::core
{

    template <typename bitmap_t, typename index_t>
    _RAFT_HOST_DEVICE inline bool bitmap_view<bitmap_t, index_t>::test(const index_t row,
                                                                       const index_t col) const
    {
        return test(row * cols_ + col);
    }

    template <typename bitmap_t, typename index_t>
    _RAFT_HOST_DEVICE void bitmap_view<bitmap_t, index_t>::set(const index_t row,
                                                               const index_t col,
                                                               bool          new_value) const
    {
        set(row * cols_ + col, &new_value);
    }

} // end namespace raft::core
