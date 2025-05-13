// Copyright (c) 2024, NVIDIA CORPORATION.
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

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.hpp>
#endif

#ifdef __HIP_PLATFORM_AMD__
#include <raft/library_types.h>
#else
#include <library_types.h>
#endif

#include <cstdint>

namespace raft
{

    template <typename T>
    constexpr auto get_cuda_data_type() -> cudaDataType_t;

    template <>
    inline constexpr auto get_cuda_data_type<int8_t>() -> cudaDataType_t
    {
        return CUDA_R_8I;
    }
    template <>
    inline constexpr auto get_cuda_data_type<uint8_t>() -> cudaDataType_t
    {
        return CUDA_R_8U;
    }
    template <>
    inline constexpr auto get_cuda_data_type<int16_t>() -> cudaDataType_t
    {
        return CUDA_R_16I;
    }
    template <>
    inline constexpr auto get_cuda_data_type<uint16_t>() -> cudaDataType_t
    {
        return CUDA_R_16U;
    }
    template <>
    inline constexpr auto get_cuda_data_type<int32_t>() -> cudaDataType_t
    {
        return CUDA_R_32I;
    }
    template <>
    inline constexpr auto get_cuda_data_type<uint32_t>() -> cudaDataType_t
    {
        return CUDA_R_32U;
    }
    template <>
    inline constexpr auto get_cuda_data_type<int64_t>() -> cudaDataType_t
    {
        return CUDA_R_64I;
    }
    template <>
    inline constexpr auto get_cuda_data_type<uint64_t>() -> cudaDataType_t
    {
        return CUDA_R_64U;
    }
    template <>
    inline constexpr auto get_cuda_data_type<half>() -> cudaDataType_t
    {
        return CUDA_R_16F;
    }
    template <>
    inline constexpr auto get_cuda_data_type<float>() -> cudaDataType_t
    {
        return CUDA_R_32F;
    }
    template <>
    inline constexpr auto get_cuda_data_type<double>() -> cudaDataType_t
    {
        return CUDA_R_64F;
    }
} // namespace raft
