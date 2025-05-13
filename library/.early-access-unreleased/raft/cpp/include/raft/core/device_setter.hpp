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
#include <raft/core/logger.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cuda_runtime.h>
#else
#include <cuda_runtime_api.h>
#endif
namespace raft
{

    /**
 * @brief A scoped setter for the active CUDA device
 *
 * On construction, the device_setter will set the active CUDA device to the
 * indicated value. On deletion, the active CUDA device will be set back to
 * its previous value. If the call to set the new active device fails, an
 * exception will be thrown. If the call to set the device back to its
 * previously selected value throws, an error will be logged, but no
 * exception will be thrown.
 *
 * @param int device_id The ID of the CUDA device to make active
 *
 */
    struct device_setter
    {
        /**
   * Return the id of the current device as an integer
   */
        static auto get_current_device()
        {
            auto result = int{};
            RAFT_CUDA_TRY(cudaGetDevice(&result));
            return result;
        }
        /**
   * Return the count of currently available CUDA devices
   */
        static auto get_device_count()
        {
            auto result = int{};
            RAFT_CUDA_TRY(cudaGetDeviceCount(&result));
            return result;
        }

        explicit device_setter(int new_device)
            : prev_device_{get_current_device()}
        {
            RAFT_CUDA_TRY(cudaSetDevice(new_device));
        }
        ~device_setter()
        {
            RAFT_CUDA_TRY_NO_THROW(cudaSetDevice(prev_device_));
        }

    private:
        int prev_device_;
    };

} // namespace raft
