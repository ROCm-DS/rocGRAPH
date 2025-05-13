// Copyright (c) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/*
 * Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#pragma once

#include <raft/core/detail/macros.hpp>

#ifdef _RAFT_HAS_CUDA
#include <raft/util/cuda_utils.cuh> // raft::shfl_xor

#ifdef __HIP_PLATFORM_AMD__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#endif
#endif
namespace raft
{
    /**
 * \brief A key identifier paired with a corresponding value
 *
 */
    template <typename _Key, typename _Value>
    struct KeyValuePair
    {
        typedef _Key   Key; ///< Key data type
        typedef _Value Value; ///< Value data type

        Key   key; ///< Item key
        Value value; ///< Item value

        /// Constructor
        KeyValuePair() = default;

#ifdef _RAFT_HAS_CUDA
        /// Conversion Constructor to allow integration w/ cub
        RAFT_INLINE_FUNCTION KeyValuePair(cub::KeyValuePair<_Key, _Value> kvp)
            : key(kvp.key)
            , value(kvp.value)
        {
        }

        RAFT_INLINE_FUNCTION operator cub::KeyValuePair<_Key, _Value>()
        {
            return cub::KeyValuePair<_Key, _Value>(key, value);
        }
#endif

        /// Constructor
        RAFT_INLINE_FUNCTION KeyValuePair(Key const& key, Value const& value)
            : key(key)
            , value(value)
        {
        }

        /// Inequality operator
        RAFT_INLINE_FUNCTION bool operator!=(const KeyValuePair& b)
        {
            return (value != b.value) || (key != b.key);
        }

        RAFT_INLINE_FUNCTION bool operator<(const KeyValuePair<_Key, _Value>& b) const
        {
            return (key < b.key) || ((key == b.key) && value < b.value);
        }

        RAFT_INLINE_FUNCTION bool operator>(const KeyValuePair<_Key, _Value>& b) const
        {
            return (key > b.key) || ((key == b.key) && value > b.value);
        }
    };

#ifdef _RAFT_HAS_CUDA
    template <typename _Key, typename _Value>
    RAFT_INLINE_FUNCTION KeyValuePair<_Key, _Value>
                         shfl_xor(const KeyValuePair<_Key, _Value>& input,
                                  int                               laneMask,
                                  int                               width = WarpSize,
                                  bitmask_type                      mask  = LANE_MASK_ALL)
    {
        return KeyValuePair<_Key, _Value>(shfl_xor(input.key, laneMask, width, mask),
                                          shfl_xor(input.value, laneMask, width, mask));
    }
#endif
} // end namespace raft
