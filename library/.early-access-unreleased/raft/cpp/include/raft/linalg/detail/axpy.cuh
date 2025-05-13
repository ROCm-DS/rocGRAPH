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

#include "cublas_wrappers.hpp"

#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resources.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cublas_v2.h>
#else
#include <cublas_v2.h>
#endif

namespace raft::linalg::detail
{

    template <typename T, bool DevicePointerMode = false>
    void axpy(raft::resources const& handle,
              const int              n,
              const T*               alpha,
              const T*               x,
              const int              incx,
              T*                     y,
              const int              incy,
              cudaStream_t           stream)
    {
        auto cublas_h = resource::get_cublas_handle(handle);
        cublas_device_pointer_mode<DevicePointerMode> pmode(cublas_h);
        RAFT_CUBLAS_TRY(cublasaxpy(cublas_h, n, alpha, x, incx, y, incy, stream));
    }

} // namespace raft::linalg::detail
