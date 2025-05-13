// Copyright (c) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <raft/core/device_span.hpp>
#include <raft/util/cuda_dev_essentials.cuh> // DI

#include <hip/hip_fp16.h>

#include <cstdint> // uintX_t

namespace raft
{

    /**
 * @defgroup SmemStores Shared memory store operations
 * @{
 * @brief Stores to shared memory (LDS)
 *
 * @param[in] addr  memory address
 * @param[in]  x    data to be stored at this address
 */
    DI void sts(uint8_t* addr, const uint8_t& x)
    {
        uint32_t offset = static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(addr)); // Convert pointer to LDS offset
        uint32_t x_int = static_cast<uint32_t>(x); // Promote to uint32_t
        asm volatile("ds_write_b8 %0, %1  \n \t"
                     : // Outputs
                     : "v"(offset), "v"(x_int) // Inputs
        );
    }
    DI void sts(uint8_t* addr, const uint8_t (&x)[1])
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        uint32_t x_int  = static_cast<uint32_t>(x[0]);
        asm volatile("ds_write_b8 %0, %1  \n \t" : : "v"(offset), "v"(x_int));
    }
    DI void sts(uint8_t* addr, const uint8_t (&x)[2])
    {
        uint32_t offset   = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        uint32_t x_int[2] = {};
        x_int[0]          = static_cast<uint32_t>(x[0]);
        x_int[1]          = static_cast<uint32_t>(x[1]);
        asm volatile("ds_write_b8 %0, %1  \n \t"
                     "ds_write_b8 %2, %3  \n \t"
                     :
                     : "v"(offset), "v"(x_int[0]), "v"(offset + 1), "v"(x_int[1]));
    }
    DI void sts(uint8_t* addr, const uint8_t (&x)[4])
    {
        uint32_t offset   = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        uint32_t x_int[4] = {};
        x_int[0]          = static_cast<uint32_t>(x[0]);
        x_int[1]          = static_cast<uint32_t>(x[1]);
        x_int[2]          = static_cast<uint32_t>(x[2]);
        x_int[3]          = static_cast<uint32_t>(x[3]);
        asm volatile("ds_write_b8 %0, %1  \n \t"
                     "ds_write_b8 %2, %3  \n \t"
                     "ds_write_b8 %4, %5  \n \t"
                     "ds_write_b8 %6, %7  \n \t"
                     :
                     : "v"(offset),
                       "v"(x_int[0]),
                       "v"(offset + 4),
                       "v"(x_int[1]),
                       "v"(offset + 8),
                       "v"(x_int[2]),
                       "v"(offset + 12),
                       "v"(x_int[3]));
    }
    DI void sts(int8_t* addr, const int8_t& x)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        int32_t  x_int  = static_cast<int32_t>(x);
        asm volatile("ds_write_b8 %0, %1  \n \t" : : "v"(offset), "v"(x_int));
    }
    DI void sts(int8_t* addr, const int8_t (&x)[1])
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        int32_t  x_int  = static_cast<int32_t>(x[0]);
        asm volatile("ds_write_b8 %0, %1  \n \t" : : "v"(offset), "v"(x_int));
    }
    DI void sts(int8_t* addr, const int8_t (&x)[2])
    {
        uint32_t offset   = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        int32_t  x_int[2] = {};
        x_int[0]          = static_cast<int32_t>(x[0]);
        x_int[1]          = static_cast<int32_t>(x[1]);
        asm volatile("ds_write_b8 %0, %1  \n \t"
                     "ds_write_b8 %2, %3  \n \t"
                     :
                     : "v"(offset), "v"(x_int[0]), "v"(offset + 1), "v"(x_int[1]));
    }
    DI void sts(int8_t* addr, const int8_t (&x)[4])
    {
        uint32_t offset   = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        int32_t  x_int[4] = {};
        x_int[0]          = static_cast<int32_t>(x[0]);
        x_int[1]          = static_cast<int32_t>(x[1]);
        x_int[2]          = static_cast<int32_t>(x[2]);
        x_int[3]          = static_cast<int32_t>(x[3]);
        asm volatile("ds_write_b8 %0, %1  \n \t"
                     "ds_write_b8 %2, %3  \n \t"
                     "ds_write_b8 %4, %5  \n \t"
                     "ds_write_b8 %6, %7  \n \t"
                     :
                     : "v"(offset),
                       "v"(x_int[0]),
                       "v"(offset + 1),
                       "v"(x_int[1]),
                       "v"(offset + 2),
                       "v"(x_int[2]),
                       "v"(offset + 3),
                       "v"(x_int[3]));
    }
    DI void sts(uint32_t* addr, const uint32_t& x)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_write_b32 %0, %1  \n \t" : : "v"(offset), "v"(x));
    }
    DI void sts(uint32_t* addr, const uint32_t (&x)[1])
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_write_b32 %0, %1  \n \t" : : "v"(offset), "v"(x[0]));
    }
    DI void sts(uint32_t* addr, const uint32_t (&x)[2])
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_write_b32 %0, %1  \n \t"
                     "ds_write_b32 %2, %3  \n \t"
                     :
                     : "v"(offset), "v"(x[0]), "v"(offset + 4), "v"(x[1]));
    }
    DI void sts(uint32_t* addr, const uint32_t (&x)[4])
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_write_b32 %0, %1  \n \t"
                     "ds_write_b32 %2, %3  \n \t"
                     "ds_write_b32 %4, %5  \n \t"
                     "ds_write_b32 %6, %7  \n \t"
                     :
                     : "v"(offset),
                       "v"(x[0]),
                       "v"(offset + 4),
                       "v"(x[1]),
                       "v"(offset + 8),
                       "v"(x[2]),
                       "v"(offset + 12),
                       "v"(x[3]));
    }
    DI void sts(int32_t* addr, const int32_t& x)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_write_b32 %0, %1  \n \t" : : "v"(offset), "v"(x));
    }
    DI void sts(int32_t* addr, const int32_t (&x)[1])
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_write_b32 %0, %1  \n \t" : : "v"(offset), "v"(x[0]));
    }
    DI void sts(int32_t* addr, const int32_t (&x)[2])
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_write_b32 %0, %1  \n \t"
                     "ds_write_b32 %2, %3  \n \t"
                     :
                     : "v"(offset), "v"(x[0]), "v"(offset + 4), "v"(x[1]));
    }
    DI void sts(int32_t* addr, const int32_t (&x)[4])
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_write_b32 %0, %1  \n \t"
                     "ds_write_b32 %2, %3  \n \t"
                     "ds_write_b32 %4, %5  \n \t"
                     "ds_write_b32 %6, %7  \n \t"
                     :
                     : "v"(offset),
                       "v"(x[0]),
                       "v"(offset + 4),
                       "v"(x[1]),
                       "v"(offset + 8),
                       "v"(x[2]),
                       "v"(offset + 12),
                       "v"(x[3]));
    }
    DI void sts(half* addr, const half& x)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        uint16_t x_int  = static_cast<const uint16_t>(x);
        asm volatile("ds_write_b16 %0, %1  \n \t" : : "v"(offset), "v"(x_int));
    }
    DI void sts(half* addr, const half (&x)[1])
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        uint16_t x_int  = static_cast<const uint16_t>(x[0]);
        asm volatile("ds_write_b16 %0, %1  \n \t" : : "v"(offset), "v"(x_int));
    }
    DI void sts(half* addr, const half (&x)[2])
    {
        uint32_t offset   = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        uint16_t x_int[2] = {};
        x_int[0]          = static_cast<uint16_t>(x[0]);
        x_int[1]          = static_cast<uint16_t>(x[1]);
        asm volatile("ds_write_b16 %0, %1  \n \t"
                     "ds_write_b16 %2, %3  \n \t"
                     :
                     : "v"(offset), "v"(x_int[0]), "v"(offset + 2), "v"(x_int[1]));
    }
    DI void sts(half* addr, const half (&x)[4])
    {
        uint32_t offset   = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        uint16_t x_int[4] = {};
        x_int[0]          = static_cast<uint16_t>(x[0]);
        x_int[1]          = static_cast<uint16_t>(x[1]);
        x_int[2]          = static_cast<uint16_t>(x[2]);
        x_int[3]          = static_cast<uint16_t>(x[3]);
        asm volatile("ds_write_b16 %0, %1  \n \t"
                     "ds_write_b16 %2, %3  \n \t"
                     "ds_write_b16 %4, %5  \n \t"
                     "ds_write_b16 %6, %7  \n \t"
                     :
                     : "v"(offset),
                       "v"(x_int[0]),
                       "v"(offset + 2),
                       "v"(x_int[1]),
                       "v"(offset + 4),
                       "v"(x_int[2]),
                       "v"(offset + 6),
                       "v"(x_int[3]));
    }
    DI void sts(half* addr, const half (&x)[8])
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        uint32_t y[4]   = {};
        y[0]            = (static_cast<uint16_t>(x[1]) << 16) | static_cast<uint16_t>(x[0]);
        y[1]            = (static_cast<uint16_t>(x[3]) << 16) | static_cast<uint16_t>(x[2]);
        y[2]            = (static_cast<uint16_t>(x[5]) << 16) | static_cast<uint16_t>(x[4]);
        y[3]            = (static_cast<uint16_t>(x[7]) << 16) | static_cast<uint16_t>(x[6]);
        asm volatile("ds_write_b32 %0, %1  \n \t"
                     "ds_write_b32 %2, %3  \n \t"
                     "ds_write_b32 %4, %5  \n \t"
                     "ds_write_b32 %6, %7  \n \t"
                     :
                     : "v"(offset),
                       "v"(y[0]),
                       "v"(offset + 4),
                       "v"(y[1]),
                       "v"(offset + 8),
                       "v"(y[2]),
                       "v"(offset + 12),
                       "v"(y[3]));
    }
    DI void sts(float* addr, const float& x)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_write_b32 %0, %1  \n \t" : : "v"(offset), "v"(x));
    }
    DI void sts(float* addr, const float (&x)[1])
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_write_b32 %0, %1  \n \t" : : "v"(offset), "v"(x[0]));
    }
    DI void sts(float* addr, const float (&x)[2])
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_write_b32 %0, %1  \n \t"
                     "ds_write_b32 %2, %3  \n \t"
                     :
                     : "v"(offset), "v"(x[0]), "v"(offset + 4), "v"(x[1]));
    }
    DI void sts(float* addr, const float (&x)[4])
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_write_b32 %0, %1  \n \t"
                     "ds_write_b32 %2, %3  \n \t"
                     "ds_write_b32 %4, %5  \n \t"
                     "ds_write_b32 %6, %7  \n \t"
                     :
                     : "v"(offset),
                       "v"(x[0]),
                       "v"(offset + 4),
                       "v"(x[1]),
                       "v"(offset + 8),
                       "v"(x[2]),
                       "v"(offset + 12),
                       "v"(x[3]));
    }
    DI void sts(double* addr, const double& x)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_write_b64 %0, %1  \n \t" : : "v"(offset), "v"(x));
    }
    DI void sts(double* addr, const double (&x)[1])
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_write_b64 %0, %1  \n \t" : : "v"(offset), "v"(x[0]));
    }
    DI void sts(double* addr, const double (&x)[2])
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_write_b64 %0, %1  \n \t"
                     "ds_write_b64 %2, %3  \n \t"
                     :
                     : "v"(offset), "v"(x[0]), "v"(offset + 8), "v"(x[1]));
    }
    /** @} */

    /**
 * @defgroup SmemLoads Shared memory load operations
 * @{
 * @brief Loads from shared memory (LDS)
 *
 * @param[out] x    the data to be loaded
 * @param[in]  addr shared memory address from where to load
 *                  (should be aligned to vector size)
 */
    DI void lds(uint8_t& x, const uint8_t* addr)
    {
        uint32_t x_int  = {};
        uint32_t offset = static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(addr)); // Convert pointer to LDS offset
        asm volatile("ds_read_u8 %0, %1  \n \t"
                     : "=v"(x_int) // Outputs
                     : "v"(offset) // Inputs
        );
        x = static_cast<uint8_t>(x_int);
    }
    DI void lds(uint8_t (&x)[1], const uint8_t* addr)
    {
        uint32_t x_int[1] = {};
        uint32_t offset   = static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(addr)); // Convert pointer to LDS offset
        asm volatile("ds_read_u8 %0, %1  \n \t"
                     : "=v"(x_int) // Outputs
                     : "v"(offset) // Inputs
        );
        x[0] = static_cast<uint8_t>(x_int[1]);
    }
    DI void lds(uint8_t (&x)[2], const uint8_t* addr)
    {
        uint32_t x_int[2] = {};
        uint32_t offset   = static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(addr)); // Convert pointer to LDS offset
        asm volatile("ds_read_u8 %0, %2  \n \t"
                     "ds_read_u8 %1, %3  \n \t"
                     : "=v"(x_int[0]), "=v"(x_int[1])
                     : "v"(offset), "v"(offset + 1));
        x[0] = static_cast<uint8_t>(x_int[0]);
        x[1] = static_cast<uint8_t>(x_int[1]);
    }
    DI void lds(uint8_t (&x)[4], const uint8_t* addr)
    {
        uint32_t x_int[4] = {};
        uint32_t offset   = static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(addr)); // Convert pointer to LDS offset
        asm volatile("ds_read_u8 %0, %4  \n \t"
                     "ds_read_u8 %1, %5  \n \t"
                     "ds_read_u8 %2, %6  \n \t"
                     "ds_read_u8 %3, %7  \n \t"
                     : "=v"(x_int[0]), "=v"(x_int[1]), "=v"(x_int[2]), "=v"(x_int[3])
                     : "v"(offset), "v"(offset + 4), "v"(offset + 8), "v"(offset + 12));
        x[0] = static_cast<uint8_t>(x_int[0]);
        x[1] = static_cast<uint8_t>(x_int[1]);
        x[2] = static_cast<uint8_t>(x_int[2]);
        x[3] = static_cast<uint8_t>(x_int[3]);
    }
    DI void lds(int8_t& x, const int8_t* addr)
    {
        int32_t  x_int  = {};
        uint32_t offset = static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(addr)); // Convert pointer to LDS offset
        asm volatile("ds_read_i8 %0, %1  \n \t"
                     : "=v"(x_int) // Outputs
                     : "v"(offset) // Inputs
        );
        x = static_cast<int8_t>(x_int);
    }
    DI void lds(int8_t (&x)[1], const int8_t* addr)
    {
        int32_t  x_int[1] = {};
        uint32_t offset   = static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(addr)); // Convert pointer to LDS offset
        asm volatile("ds_read_i8 %0, %1  \n \t"
                     : "=v"(x_int) // Outputs
                     : "v"(offset) // Inputs
        );
        x[0] = static_cast<int8_t>(x_int[0]);
    }
    DI void lds(int8_t (&x)[2], const int8_t* addr)
    {
        int32_t  x_int[2] = {};
        uint32_t offset   = static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(addr)); // Convert pointer to LDS offset
        asm volatile("ds_read_i8 %0, %2  \n \t"
                     "ds_read_i8 %1, %3  \n \t"
                     : "=v"(x_int[0]), "=v"(x_int[1])
                     : "v"(offset), "v"(offset + 1));
        x[0] = static_cast<int8_t>(x_int[0]);
        x[1] = static_cast<int8_t>(x_int[1]);
    }
    DI void lds(int8_t (&x)[4], const int8_t* addr)
    {
        int32_t  x_int[4] = {};
        uint32_t offset   = static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(addr)); // Convert pointer to LDS offset
        asm volatile("ds_read_i8 %0, %4  \n \t"
                     "ds_read_i8 %1, %5  \n \t"
                     "ds_read_i8 %2, %6  \n \t"
                     "ds_read_i8 %3, %7  \n \t"
                     : "=v"(x_int[0]), "=v"(x_int[1]), "=v"(x_int[2]), "=v"(x_int[3])
                     : "v"(offset), "v"(offset + 1), "v"(offset + 2), "v"(offset + 3));
        x[0] = static_cast<int8_t>(x_int[0]);
        x[1] = static_cast<int8_t>(x_int[1]);
        x[2] = static_cast<int8_t>(x_int[2]);
        x[3] = static_cast<int8_t>(x_int[3]);
    }
    DI void lds(uint32_t (&x)[4], const uint32_t* addr)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_read_b32 %0, %4  \n \t"
                     "ds_read_b32 %1, %5  \n \t"
                     "ds_read_b32 %2, %6  \n \t"
                     "ds_read_b32 %3, %7  \n \t"
                     : "=v"(x[0]), "=v"(x[1]), "=v"(x[2]), "=v"(x[3])
                     : "v"(offset), "v"(offset + 4), "v"(offset + 8), "v"(offset + 12));
    }
    DI void lds(uint32_t (&x)[2], const uint32_t* addr)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_read_b32 %0, %2  \n \t"
                     "ds_read_b32 %1, %3  \n \t"
                     : "=v"(x[0]), "=v"(x[1])
                     : "v"(offset), "v"(offset + 4));
    }
    DI void lds(uint32_t (&x)[1], const uint32_t* addr)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_read_b32 %0, %1  \n \t" : "=v"(x[0]) : "v"(offset));
    }
    DI void lds(uint32_t& x, const uint32_t* addr)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_read_b32 %0, %1  \n \t" : "=v"(x) : "v"(offset));
    }
    DI void lds(int32_t (&x)[4], const int32_t* addr)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_read_b32 %0, %4  \n \t"
                     "ds_read_b32 %1, %5  \n \t"
                     "ds_read_b32 %2, %6  \n \t"
                     "ds_read_b32 %3, %7  \n \t"
                     : "=v"(x[0]), "=v"(x[1]), "=v"(x[2]), "=v"(x[3])
                     : "v"(offset), "v"(offset + 4), "v"(offset + 8), "v"(offset + 12));
    }
    DI void lds(int32_t (&x)[2], const int32_t* addr)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_read_b32 %0, %2  \n \t"
                     "ds_read_b32 %1, %3  \n \t"
                     : "=v"(x[0]), "=v"(x[1])
                     : "v"(offset), "v"(offset + 4));
    }
    DI void lds(int32_t (&x)[1], const int32_t* addr)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_read_b32 %0, %1  \n \t" : "=v"(x[0]) : "v"(offset));
    }
    DI void lds(int32_t& x, const int32_t* addr)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_read_b32 %0, %1  \n \t" : "=v"(x) : "v"(offset));
    }
    DI void lds(half& x, const half* addr)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        uint16_t x_int  = {};
        asm volatile("ds_read_u16 %0, %1  \n \t" : "=v"(x_int) : "v"(offset));
        x = static_cast<half>(x_int);
    }
    DI void lds(half (&x)[1], const half* addr)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        uint16_t x_int  = {};
        asm volatile("ds_read_u16 %0, %1  \n \t" : "=v"(x_int) : "v"(offset));
        x[0] = static_cast<half>(x_int);
    }
    DI void lds(half (&x)[2], const half* addr)
    {
        uint32_t offset   = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        uint16_t x_int[2] = {};
        asm volatile("ds_read_u16 %0, %2  \n \t"
                     "ds_read_u16 %1, %3  \n \t"
                     : "=v"(x_int[0]), "=v"(x_int[1])
                     : "v"(offset), "v"(offset + 2));
        x[0] = static_cast<half>(x_int[0]);
        x[1] = static_cast<half>(x_int[1]);
    }
    DI void lds(half (&x)[4], const half* addr)
    {
        uint32_t offset   = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        uint16_t x_int[4] = {};
        asm volatile("ds_read_u16 %0, %4  \n \t"
                     "ds_read_u16 %1, %5  \n \t"
                     "ds_read_u16 %2, %6  \n \t"
                     "ds_read_u16 %3, %7  \n \t"
                     : "=v"(x_int[0]), "=v"(x_int[1]), "=v"(x_int[2]), "=v"(x_int[3])
                     : "v"(offset), "v"(offset + 2), "v"(offset + 4), "v"(offset + 6));
        x[0] = static_cast<half>(x_int[0]);
        x[1] = static_cast<half>(x_int[1]);
        x[2] = static_cast<half>(x_int[2]);
        x[3] = static_cast<half>(x_int[3]);
    }
    DI void lds(half (&x)[8], const half* addr)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        uint32_t y[4]   = {};
        asm volatile("ds_read_b32 %0, %4  \n \t"
                     "ds_read_b32 %1, %5  \n \t"
                     "ds_read_b32 %2, %6  \n \t"
                     "ds_read_b32 %3, %7  \n \t"
                     : "=v"(y[0]), "=v"(y[1]), "=v"(y[2]), "=v"(y[3])
                     : "v"(offset), "v"(offset + 4), "v"(offset + 8), "v"(offset + 12));
        x[0] = static_cast<half>(y[0] & 0xFFFF); // Extract the lower 16 bits of y[0]
        x[1] = static_cast<half>((y[0] >> 16) & 0xFFFF); // Extract the upper 16 bits of y[0]
        x[2] = static_cast<half>(y[1] & 0xFFFF); // Extract the lower 16 bits of y[1]
        x[3] = static_cast<half>((y[1] >> 16) & 0xFFFF); // Extract the upper 16 bits of y[1]
        x[4] = static_cast<half>(y[2] & 0xFFFF); // Extract the lower 16 bits of y[2]
        x[5] = static_cast<half>((y[2] >> 16) & 0xFFFF); // Extract the upper 16 bits of y[2]
        x[6] = static_cast<half>(y[3] & 0xFFFF); // Extract the lower 16 bits of y[3]
        x[7] = static_cast<half>((y[3] >> 16) & 0xFFFF); // Extract the upper 16 bits of y[3]
    }
    DI void lds(float& x, const float* addr)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_read_b32 %0, %1  \n \t" : "=v"(x) : "v"(offset));
    }
    DI void lds(float (&x)[1], const float* addr)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_read_b32 %0, %1  \n \t" : "=v"(x[0]) : "v"(offset));
    }
    DI void lds(float (&x)[2], const float* addr)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_read_b32 %0, %2  \n \t"
                     "ds_read_b32 %1, %3  \n \t"
                     : "=v"(x[0]), "=v"(x[1])
                     : "v"(offset), "v"(offset + 4));
    }
    DI void lds(float (&x)[4], const float* addr)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_read_b32 %0, %4  \n \t"
                     "ds_read_b32 %1, %5  \n \t"
                     "ds_read_b32 %2, %6  \n \t"
                     "ds_read_b32 %3, %7  \n \t"
                     : "=v"(x[0]), "=v"(x[1]), "=v"(x[2]), "=v"(x[3])
                     : "v"(offset), "v"(offset + 4), "v"(offset + 8), "v"(offset + 12));
    }
    DI void lds(float& x, float* addr)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_read_b32 %0, %1  \n \t" : "=v"(x) : "v"(offset));
    }
    DI void lds(float (&x)[1], float* addr)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_read_b32 %0, %1  \n \t" : "=v"(x[0]) : "v"(offset));
    }
    DI void lds(float (&x)[2], float* addr)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_read_b32 %0, %2  \n \t"
                     "ds_read_b32 %1, %3  \n \t"
                     : "=v"(x[0]), "=v"(x[1])
                     : "v"(offset), "v"(offset + 4));
    }
    DI void lds(float (&x)[4], float* addr)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_read_b32 %0, %4  \n \t"
                     "ds_read_b32 %1, %5  \n \t"
                     "ds_read_b32 %2, %6  \n \t"
                     "ds_read_b32 %3, %7  \n \t"
                     : "=v"(x[0]), "=v"(x[1]), "=v"(x[2]), "=v"(x[3])
                     : "v"(offset), "v"(offset + 4), "v"(offset + 8), "v"(offset + 12));
    }
    DI void lds(double& x, double* addr)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_read_b64 %0, %1  \n \t" : "=v"(x) : "v"(offset));
    }
    DI void lds(double (&x)[1], double* addr)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_read_b64 %0, %1  \n \t" : "=v"(x[0]) : "v"(offset));
    }
    DI void lds(double (&x)[2], double* addr)
    {
        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
        asm volatile("ds_read_b64 %0, %2  \n \t"
                     "ds_read_b64 %1, %3  \n \t"
                     : "=v"(x[0]), "=v"(x[1])
                     : "v"(offset), "v"(offset + 8));
    }
    /** @} */

    /**
 * @defgroup GlobalLoads Global cached load operations
 * @{
 * @brief Load from global memory with caching(glc)
 * @param[out] x    data to be loaded from global memory
 * @param[in]  addr address in global memory from where to load
 */
    DI void ldg(float& x, const float* addr)
    {
        asm volatile("global_load_dword %0 %1 off glc \n \t" : "=v"(x) : "v"(addr));
    }
    DI void ldg(float (&x)[1], const float* addr)
    {
        asm volatile("global_load_dword %0 %1 off glc \n \t" : "=v"(x[0]) : "v"(addr));
    }
    DI void ldg(float (&x)[2], const float* addr)
    {
        asm volatile("global_load_dword %0 %2 off glc \n \t"
                     "global_load_dword %1 %3 off glc \n \t"
                     : "=v"(x[0]), "=v"(x[1])
                     : "v"(addr), "v"(addr + 1));
    }
    DI void ldg(float (&x)[4], const float* addr)
    {
        asm volatile("global_load_dword %0, %4 off glc \n \t"
                     "global_load_dword %1, %5 off glc \n \t"
                     "global_load_dword %2, %6 off glc \n \t"
                     "global_load_dword %3, %7 off glc \n \t"
                     : "=v"(x[0]), "=v"(x[1]), "=v"(x[2]), "=v"(x[3])
                     : "v"(addr), "v"(addr + 1), "v"(addr + 2), "v"(addr + 3));
    }
    DI void ldg(half& x, const half* addr)
    {
        asm volatile("global_load_ushort %0 %1 off glc \n \t"
                     : "=v"(*reinterpret_cast<uint16_t*>(&x))
                     : "v"(addr));
    }
    DI void ldg(half (&x)[1], const half* addr)
    {
        asm volatile("global_load_ushort %0 %1 off glc \n \t"
                     : "=v"(*reinterpret_cast<uint16_t*>(x))
                     : "v"(addr));
    }
    DI void ldg(half (&x)[2], const half* addr)
    {
        asm volatile("global_load_ushort %0 %1 off glc \n \t"
                     "global_load_ushort %1 %3 off glc \n \t"
                     : "=v"(*reinterpret_cast<uint16_t*>(x)),
                       "=v"(*reinterpret_cast<uint16_t*>(x + 1))
                     : "v"(addr), "v"(addr + 1));
    }
    DI void ldg(half (&x)[4], const half* addr)
    {
        asm volatile("global_load_ushort %0 %4 off glc \n \t"
                     "global_load_ushort %1 %5 off glc \n \t"
                     "global_load_ushort %2, %6 off glc \n \t"
                     "global_load_ushort %3, %7 off glc \n \t"
                     : "=v"(*reinterpret_cast<uint16_t*>(x)),
                       "=v"(*reinterpret_cast<uint16_t*>(x + 1)),
                       "=v"(*reinterpret_cast<uint16_t*>(x + 2)),
                       "=v"(*reinterpret_cast<uint16_t*>(x + 3))
                     : "v"(addr), "v"(addr + 1), "v"(addr + 2), "v"(addr + 3));
    }

    DI void ldg(half (&x)[8], const half* addr)
    {
        asm volatile("global_load_ushort %0 %8 off glc \n \t"
                     "global_load_ushort %1 %9 off glc \n \t"
                     "global_load_ushort %2 %10 off glc \n \t"
                     "global_load_ushort %3, %11 off glc \n \t"
                     "global_load_ushort %4 %12 off glc \n \t"
                     "global_load_ushort %5 %13 off glc \n \t"
                     "global_load_ushort %6, %14 off glc \n \t"
                     "global_load_ushort %7, %15 off glc \n \t"
                     : "=v"(*reinterpret_cast<uint16_t*>(x)),
                       "=v"(*reinterpret_cast<uint16_t*>(x + 1)),
                       "=v"(*reinterpret_cast<uint16_t*>(x + 2)),
                       "=v"(*reinterpret_cast<uint16_t*>(x + 3)),
                       "=v"(*reinterpret_cast<uint16_t*>(x + 4)),
                       "=v"(*reinterpret_cast<uint16_t*>(x + 5)),
                       "=v"(*reinterpret_cast<uint16_t*>(x + 6)),
                       "=v"(*reinterpret_cast<uint16_t*>(x + 7))
                     : "v"(addr),
                       "v"(addr + 1),
                       "v"(addr + 2),
                       "v"(addr + 3),
                       "v"(addr + 4),
                       "v"(addr + 5),
                       "v"(addr + 6),
                       "v"(addr + 7));
    }
    DI void ldg(double& x, const double* addr)
    {
        asm volatile("global_load_dwordx2 %0 %1 off glc \n \t" : "=v"(x) : "v"(addr));
    }
    DI void ldg(double (&x)[1], const double* addr)
    {
        asm volatile("global_load_dwordx2 %0 %1 off glc \n \t" : "=v"(x[0]) : "v"(addr));
    }
    DI void ldg(double (&x)[2], const double* addr)
    {
        asm volatile("global_load_dwordx2 %0 %2 off glc \n \t"
                     "global_load_dwordx2 %1 %3 off glc \n \t"
                     : "=v"(x[0]), "=v"(x[1])
                     : "v"(addr), "v"(addr + 1));
    }
    DI void ldg(uint32_t& x, const uint32_t* const& addr)
    {
        asm volatile("global_load_dword %0 %1 off glc \n \t" : "=v"(x) : "v"(addr));
    }
    DI void ldg(uint32_t (&x)[1], const uint32_t* const& addr)
    {
        asm volatile("global_load_dword %0 %1 off glc \n \t" : "=v"(x[0]) : "v"(addr));
    }
    DI void ldg(uint32_t (&x)[2], const uint32_t* const& addr)
    {
        asm volatile("global_load_dword %0 %2 off glc \n \t"
                     "global_load_dword %1 %3 off glc \n \t"
                     : "=v"(x[0]), "=v"(x[1])
                     : "v"(addr), "v"(addr + 1));
    }

    DI void ldg(uint32_t (&x)[4], const uint32_t* const& addr)
    {
        asm volatile("global_load_dword %0, %4 off glc \n \t"
                     "global_load_dword %1, %5 off glc \n \t"
                     "global_load_dword %2, %6 off glc \n \t"
                     "global_load_dword %3, %7 off glc \n \t"
                     : "=v"(x[0]), "=v"(x[1]), "=v"(x[2]), "=v"(x[3])
                     : "v"(addr), "v"(addr + 1), "v"(addr + 2), "v"(addr + 3));
    }
    DI void ldg(int32_t& x, const int32_t* const& addr)
    {
        asm volatile("global_load_dword %0 %1 off glc \n \t" : "=v"(x) : "v"(addr));
    }

    DI void ldg(int32_t (&x)[1], const int32_t* const& addr)
    {
        asm volatile("global_load_dword %0 %1 off glc \n \t" : "=v"(x[0]) : "v"(addr));
    }

    DI void ldg(int32_t (&x)[2], const int32_t* const& addr)
    {
        asm volatile("global_load_dword %0 %2 off glc \n \t"
                     "global_load_dword %1 %3 off glc \n \t"
                     : "=v"(x[0]), "=v"(x[1])
                     : "v"(addr), "v"(addr + 1));
    }

    DI void ldg(int32_t (&x)[4], const int32_t* const& addr)
    {
        asm volatile("global_load_dword %0, %4 off glc \n \t"
                     "global_load_dword %1, %5 off glc \n \t"
                     "global_load_dword %2, %6 off glc \n \t"
                     "global_load_dword %3, %7 off glc \n \t"
                     : "=v"(x[0]), "=v"(x[1]), "=v"(x[2]), "=v"(x[3])
                     : "v"(addr), "v"(addr + 1), "v"(addr + 2), "v"(addr + 3));
    }

    DI void ldg(uint8_t& x, const uint8_t* const& addr)
    {
        uint32_t raw_val = {};
        asm volatile("global_load_ubyte %0 %1 off glc \n \t" : "=v"(raw_val) : "v"(addr));
        x = static_cast<uint8_t>(raw_val);
    }

    DI void ldg(uint8_t (&x)[1], const uint8_t* const& addr)
    {
        uint32_t raw_val = {};
        asm volatile("global_load_ubyte %0 %1 off glc \n \t" : "=v"(raw_val) : "v"(addr));
        x[0] = static_cast<uint8_t>(raw_val);
    }

    DI void ldg(uint8_t (&x)[2], const uint8_t* const& addr)
    {
        uint32_t raw_val[2] = {};
        asm volatile("global_load_ubyte %0 %2 off glc \n \t"
                     "global_load_ubyte %1 %3 off glc \n \t"
                     : "=v"(raw_val[0]), "=v"(raw_val[1])
                     : "v"(addr), "v"(addr + 1));
        x[0] = static_cast<uint8_t>(raw_val[0]);
        x[1] = static_cast<uint8_t>(raw_val[1]);
    }

    DI void ldg(uint8_t (&x)[4], const uint8_t* const& addr)
    {
        uint32_t raw_val[4] = {};
        asm volatile("global_load_ubyte %0, %4 off glc \n \t"
                     "global_load_ubyte %1, %5 off glc \n \t"
                     "global_load_ubyte %2, %6 off glc \n \t"
                     "global_load_ubyte %3, %7 off glc \n \t"
                     : "=v"(raw_val[0]), "=v"(raw_val[1]), "=v"(raw_val[2]), "=v"(raw_val[3])
                     : "v"(addr), "v"(addr + 1), "v"(addr + 2), "v"(addr + 3));
        x[0] = static_cast<uint8_t>(raw_val[0]);
        x[1] = static_cast<uint8_t>(raw_val[1]);
        x[2] = static_cast<uint8_t>(raw_val[2]);
        x[3] = static_cast<uint8_t>(raw_val[3]);
    }

    DI void ldg(int8_t& x, const int8_t* const& addr)
    {
        int32_t raw_val = {};
        asm volatile("global_load_sbyte %0 %1 off glc \n \t" : "=v"(x) : "v"(addr));
        x = static_cast<int8_t>(raw_val);
    }

    DI void ldg(int8_t (&x)[1], const int8_t* const& addr)
    {
        int32_t raw_val = {};
        asm volatile("global_load_sbyte %0 %1 off glc \n \t" : "=v"(x[0]) : "v"(addr));
        x[0] = static_cast<int8_t>(raw_val);
    }

    DI void ldg(int8_t (&x)[2], const int8_t* const& addr)
    {
        int32_t raw_val[2] = {};
        asm volatile("global_load_sbyte %0 %2 off glc \n \t"
                     "global_load_sbyte %1 %3 off glc \n \t"
                     : "=v"(raw_val[0]), "=v"(raw_val[1])
                     : "v"(addr), "v"(addr + 1));
        x[0] = static_cast<int8_t>(raw_val[0]);
        x[1] = static_cast<int8_t>(raw_val[1]);
    }

    DI void ldg(int8_t (&x)[4], const int8_t* const& addr)
    {
        int32_t raw_val[4] = {};
        asm volatile("global_load_sbyte %0, %4 off glc \n \t"
                     "global_load_sbyte %1, %5 off glc \n \t"
                     "global_load_sbyte %2, %6 off glc \n \t"
                     "global_load_sbyte %3, %7 off glc \n \t"
                     : "=v"(raw_val[0]), "=v"(raw_val[1]), "=v"(raw_val[2]), "=v"(raw_val[3])
                     : "v"(addr), "v"(addr + 1), "v"(addr + 2), "v"(addr + 3));
        x[0] = static_cast<int8_t>(raw_val[0]);
        x[1] = static_cast<int8_t>(raw_val[1]);
        x[2] = static_cast<int8_t>(raw_val[2]);
        x[3] = static_cast<int8_t>(raw_val[3]);
    }

    /**
 * @brief Executes a 1D block strided copy
 * @param dst destination pointer
 * @param src source pointer
 * @param size number of items to copy
 */
    template <typename T>
    DI void block_copy(T* dst, const T* src, const size_t size)
    {
        for(auto i = threadIdx.x; i < size; i += blockDim.x)
        {
            dst[i] = src[i];
        }
    }

    /**
 * @brief Executes a 1D block strided copy
 * @param dst span of destination pointer
 * @param src span of source pointer
 * @param size number of items to copy
 */
    template <typename T>
    DI void block_copy(raft::device_span<T>             dst,
                       const raft::device_span<const T> src,
                       const size_t                     size)
    {
        assert(src.size() >= size);
        assert(dst.size() >= size);
        block_copy(dst.data(), src.data(), size);
    }

    /**
 * @brief Executes a 1D block strided copy
 * @param dst span of destination pointer
 * @param src span of source pointer
 * @param size number of items to copy
 */
    template <typename T>
    DI void block_copy(raft::device_span<T> dst, const raft::device_span<T> src, const size_t size)
    {
        assert(src.size() >= size);
        assert(dst.size() >= size);
        block_copy(dst.data(), src.data(), size);
    }

    /**
 * @brief Executes a 1D block strided copy
 * @param dst span of destination pointer
 * @param src span of source pointer
 */
    template <typename T>
    DI void block_copy(raft::device_span<T> dst, const raft::device_span<T> src)
    {
        assert(dst.size() >= src.size());
        block_copy(dst, src, src.size());
    }

    /** @} */

} // namespace raft
