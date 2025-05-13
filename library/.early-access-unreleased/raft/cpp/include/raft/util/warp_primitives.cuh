// Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <raft/core/cudart_utils.hpp>
#include <raft/core/operators.hpp>
#include <raft/util/bitwise_operations.hpp>
#include <raft/util/cuda_dev_essentials.cuh>

#ifdef __HIP_PLATFORM_AMD__
using bitmask_type = uint64_t;
#undef CUDART_VERSION
// Force "CUDART_VERSION" to be greater than 9000 when compiling using the HIP/AMD toolchain to
// force the  selection of the "*_sync" version of warp-primitives as opposed to the deprecated
// non-sync versions.
#define CUDART_VERSION 14000
#else
using bitmask_type = uint32_t;
#endif

#include <stdint.h>

namespace raft
{

    /**
 * \return the full mask: all bits are set to 1.
 */
    __device__ inline constexpr bitmask_type LANE_MASK_ALL = ~0;

    /** True CUDA alignment of a type (adapted from CUB) */
    template <typename T>
    struct cuda_alignment
    {
        struct Pad
        {
            T    val;
            char byte;
        };

        static constexpr int bytes = sizeof(Pad) - sizeof(T);
    };

    template <typename LargeT, typename UnitT>
    struct is_multiple
    {
        static constexpr int  large_align_bytes = cuda_alignment<LargeT>::bytes;
        static constexpr int  unit_align_bytes  = cuda_alignment<UnitT>::bytes;
        static constexpr bool value
            = (sizeof(LargeT) % sizeof(UnitT) == 0) && (large_align_bytes % unit_align_bytes == 0);
    };

    template <typename LargeT, typename UnitT>
    inline constexpr bool is_multiple_v = is_multiple<LargeT, UnitT>::value;

    /** apply a warp-wide fence (useful from Volta+ archs) */
    DI void warpFence()
    {
        hip_warp_primitives::__syncwarp();
    }

    /** warp-wide any boolean aggregator */
    DI bool any(bool inFlag, bitmask_type mask = LANE_MASK_ALL)
    {
        inFlag = hip_warp_primitives::__any_sync(mask, inFlag);
        return inFlag;
    }

    /** warp-wide all boolean aggregator */
    DI bool all(bool inFlag, bitmask_type mask = LANE_MASK_ALL)
    {
        inFlag = hip_warp_primitives::__all_sync(mask, inFlag);
        return inFlag;
    }

    /** For every thread in the warp, set the corresponding bit to the thread's flag value.  */
    DI auto ballot(bool inFlag, bitmask_type mask = LANE_MASK_ALL)
    {
        return hip_warp_primitives::__ballot_sync(mask, inFlag);
    }

    template <typename T>
    struct is_shuffleable
    {
        static constexpr bool value = std::is_same_v<T, int> || std::is_same_v<T, unsigned int>
                                      || std::is_same_v<T, long> || std::is_same_v<T, unsigned long>
                                      || std::is_same_v<T, long long>
                                      || std::is_same_v<T, unsigned long long>
                                      || std::is_same_v<T, float> || std::is_same_v<T, double>;
    };

    template <typename T>
    inline constexpr bool is_shuffleable_v = is_shuffleable<T>::value;

    /**
 * @brief Shuffle the data inside a warp
 * @tparam T the data type
 * @param val value to be shuffled
 * @param srcLane lane from where to shuffle
 * @param width lane width
 * @param mask mask of participating threads (Volta+)
 * @return the shuffled data
 */
    template <typename T>
    DI  std::enable_if_t<is_shuffleable_v<T>, T>
        shfl(T val, int srcLane, int width = WarpSize, bitmask_type mask = LANE_MASK_ALL)
    {
        return hip_warp_primitives::__shfl_sync(mask, val, srcLane, width);
    }

    /// Overload of shfl for data types not supported by the CUDA intrinsics
    template <typename T>
    DI  std::enable_if_t<!is_shuffleable_v<T>, T>
        shfl(T val, int srcLane, int width = WarpSize, bitmask_type mask = LANE_MASK_ALL)
    {
        using UnitT = std::conditional_t<
            is_multiple_v<T, int>,
            unsigned int,
            std::conditional_t<is_multiple_v<T, short>, unsigned short, unsigned char>>;

        constexpr int n_words = sizeof(T) / sizeof(UnitT);

        T      output;
        UnitT* output_alias = reinterpret_cast<UnitT*>(&output);
        UnitT* input_alias  = reinterpret_cast<UnitT*>(&val);

        unsigned int shuffle_word;
        shuffle_word    = shfl((unsigned int)input_alias[0], srcLane, width, mask);
        output_alias[0] = shuffle_word;

#pragma unroll
        for(int i = 1; i < n_words; ++i)
        {
            shuffle_word    = shfl((unsigned int)input_alias[i], srcLane, width, mask);
            output_alias[i] = shuffle_word;
        }

        return output;
    }

    /**
 * @brief Shuffle the data inside a warp from lower lane IDs
 * @tparam T the data type
 * @param val value to be shuffled
 * @param delta lower lane ID delta from where to shuffle
 * @param width lane width
 * @param mask mask of participating threads (Volta+)
 * @return the shuffled data
 */
    template <typename T>
    DI  std::enable_if_t<is_shuffleable_v<T>, T>
        shfl_up(T val, int delta, int width = WarpSize, bitmask_type mask = LANE_MASK_ALL)
    {
        return hip_warp_primitives::__shfl_up_sync(mask, val, delta, width);
    }

    /// Overload of shfl_up for data types not supported by the CUDA intrinsics
    template <typename T>
    DI  std::enable_if_t<!is_shuffleable_v<T>, T>
        shfl_up(T val, int delta, int width = WarpSize, bitmask_type mask = LANE_MASK_ALL)
    {
        using UnitT = std::conditional_t<
            is_multiple_v<T, int>,
            unsigned int,
            std::conditional_t<is_multiple_v<T, short>, unsigned short, unsigned char>>;

        constexpr int n_words = sizeof(T) / sizeof(UnitT);

        T      output;
        UnitT* output_alias = reinterpret_cast<UnitT*>(&output);
        UnitT* input_alias  = reinterpret_cast<UnitT*>(&val);

        unsigned int shuffle_word;
        shuffle_word    = shfl_up((unsigned int)input_alias[0], delta, width, mask);
        output_alias[0] = shuffle_word;

#pragma unroll
        for(int i = 1; i < n_words; ++i)
        {
            shuffle_word    = shfl_up((unsigned int)input_alias[i], delta, width, mask);
            output_alias[i] = shuffle_word;
        }

        return output;
    }

    /**
 * @brief Shuffle the data inside a warp
 * @tparam T the data type
 * @param val value to be shuffled
 * @param laneMask mask to be applied in order to perform xor shuffle
 * @param width lane width
 * @param mask mask of participating threads (Volta+)
 * @return the shuffled data
 */
    template <typename T>
    DI  std::enable_if_t<is_shuffleable_v<T>, T>
        shfl_xor(T val, int laneMask, int width = WarpSize, bitmask_type mask = LANE_MASK_ALL)
    {
        return hip_warp_primitives::__shfl_xor_sync(mask, val, laneMask, width);
    }

    /// Overload of shfl_xor for data types not supported by the CUDA intrinsics
    template <typename T>
    DI  std::enable_if_t<!is_shuffleable_v<T>, T>
        shfl_xor(T val, int laneMask, int width = WarpSize, bitmask_type mask = LANE_MASK_ALL)
    {
        using UnitT = std::conditional_t<
            is_multiple_v<T, int>,
            unsigned int,
            std::conditional_t<is_multiple_v<T, short>, unsigned short, unsigned char>>;

        constexpr int n_words = sizeof(T) / sizeof(UnitT);

        T      output;
        UnitT* output_alias = reinterpret_cast<UnitT*>(&output);
        UnitT* input_alias  = reinterpret_cast<UnitT*>(&val);

        unsigned int shuffle_word;
        shuffle_word    = shfl_xor((unsigned int)input_alias[0], laneMask, width, mask);
        output_alias[0] = shuffle_word;

#pragma unroll
        for(int i = 1; i < n_words; ++i)
        {
            shuffle_word    = shfl_xor((unsigned int)input_alias[i], laneMask, width, mask);
            output_alias[i] = shuffle_word;
        }

        return output;
    }
#ifdef __HIP_PLATFORM_AMD__
// Undefine it to not affect any other source files
#undef CUDART_VERSION
#endif
} // namespace raft
