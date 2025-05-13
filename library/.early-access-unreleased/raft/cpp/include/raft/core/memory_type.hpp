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
#include <cstdint>
#include <optional>
#ifndef RAFT_DISABLE_CUDA
#include <raft/util/cuda_rt_essentials.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include <type_traits>
#else
#include <raft/core/logger.hpp>
#endif

namespace raft
{
    enum class memory_type : std::uint8_t
    {
        host    = std::uint8_t{0},
        pinned  = std::uint8_t{1},
        device  = std::uint8_t{2},
        managed = std::uint8_t{3}
    };

    auto constexpr is_device_accessible(memory_type mem_type)
    {
        return (mem_type == memory_type::device || mem_type == memory_type::managed
                || mem_type == memory_type::pinned);
    }
    auto constexpr is_host_accessible(memory_type mem_type)
    {
        return (mem_type == memory_type::host || mem_type == memory_type::managed
                || mem_type == memory_type::pinned);
    }
    auto constexpr is_host_device_accessible(memory_type mem_type)
    {
        return is_device_accessible(mem_type) && is_host_accessible(mem_type);
    }

    auto constexpr has_compatible_accessibility(memory_type old_mem_type, memory_type new_mem_type)
    {
        return ((!is_device_accessible(new_mem_type) || is_device_accessible(old_mem_type))
                && (!is_host_accessible(new_mem_type) || is_host_accessible(old_mem_type)));
    }

    template <memory_type... mem_types>
    struct memory_type_constant
    {
        static_assert(sizeof...(mem_types) < 2, "At most one memory type can be specified");
        auto static constexpr value = []() {
            auto result = std::optional<memory_type>{};
            if constexpr(sizeof...(mem_types) == 1)
            {
                result = std::make_optional(mem_types...);
            }
            return result;
        }();
    };

    namespace detail
    {

        template <bool is_host_accessible, bool is_device_accessible>
        auto constexpr memory_type_from_access()
        {
            if constexpr(is_host_accessible && is_device_accessible)
            {
                return memory_type::managed;
            }
            else if constexpr(is_host_accessible)
            {
                return memory_type::host;
            }
            else if constexpr(is_device_accessible)
            {
                return memory_type::device;
            }
            static_assert(is_host_accessible || is_device_accessible,
                          "Must be either host or device accessible to return a valid memory type");
        }

    } // end namespace detail

    template <typename T>
    auto memory_type_from_pointer(T* ptr)
    {
        auto result = memory_type::host;
#ifndef RAFT_DISABLE_CUDA
// Special treatment of nullptr on HIP/AMD:
// In contrast to cuda, hipPointerGetAttributes
// currently (ROCm 6.1.2) fails if a nullptr is passed on the host.
// We mimic CUDA's behavior by returning memory_type::host.
#ifdef __HIP_PLATFORM_AMD__
        if(!ptr)
        {
            return result;
        }
#endif

        auto attrs = cudaPointerAttributes{};
        RAFT_CUDA_TRY(cudaPointerGetAttributes(&attrs, ptr));
        switch(attrs.type)
        {
        case cudaMemoryTypeDevice:
            result = memory_type::device;
            break;
        case cudaMemoryTypeHost:
            result = memory_type::host;
            break;
        case cudaMemoryTypeManaged:
            result = memory_type::managed;
            break;
        default:
            result = memory_type::host;
        }
#else
        RAFT_LOG_DEBUG("RAFT compiled without CUDA support, assuming pointer is host pointer");
#endif
        return result;
    }
} // end namespace raft
