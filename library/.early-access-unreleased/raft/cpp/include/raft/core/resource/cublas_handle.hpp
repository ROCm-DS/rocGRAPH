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

#include <raft/core/cublas_macros.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cublas_v2.h>
#else
#include <cublas_v2.h>
#endif

namespace raft::resource
{

    class cublas_resource : public resource
    {
    public:
        cublas_resource(rmm::cuda_stream_view stream)
        {
            RAFT_CUBLAS_TRY_NO_THROW(cublasCreate(&cublas_res));
            RAFT_CUBLAS_TRY_NO_THROW(cublasSetStream(cublas_res, stream));
        }

        ~cublas_resource() override
        {
            RAFT_CUBLAS_TRY_NO_THROW(cublasDestroy(cublas_res));
        }

        void* get_resource() override
        {
            return &cublas_res;
        }

    private:
        cublasHandle_t cublas_res;
    };

    /**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
    class cublas_resource_factory : public resource_factory
    {
    public:
        cublas_resource_factory(rmm::cuda_stream_view stream)
            : stream_(stream)
        {
        }
        resource_type get_resource_type() override
        {
            return resource_type::CUBLAS_HANDLE;
        }
        resource* make_resource() override
        {
            return new cublas_resource(stream_);
        }

    private:
        rmm::cuda_stream_view stream_;
    };

    /**
 * @defgroup resource_cublas cuBLAS handle resource functions
 * @{
 */

    /**
 * Load a `cublasHandle_t` from raft res if it exists, otherwise add it and return it.
 *
 * @param[in] res the raft resources object
 * @return cublas handle
 */
    inline cublasHandle_t get_cublas_handle(resources const& res)
    {
        if(!res.has_resource_factory(resource_type::CUBLAS_HANDLE))
        {
            cudaStream_t stream = get_cuda_stream(res);
            res.add_resource_factory(std::make_shared<cublas_resource_factory>(stream));
        }
        auto ret = *res.get_resource<cublasHandle_t>(resource_type::CUBLAS_HANDLE);
        RAFT_CUBLAS_TRY(cublasSetStream(ret, get_cuda_stream(res)));
        return ret;
    };

    /**
 * @}
 */

} // namespace raft::resource
