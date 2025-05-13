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

#include <raft/core/cusolver_macros.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cusolver.h>
#else
#include <cusolverSp.h>
#endif

namespace raft::resource
{

    /**
 *
 */
    class cusolver_sp_resource : public resource
    {
    public:
        cusolver_sp_resource(rmm::cuda_stream_view stream)
        {
            RAFT_CUSOLVER_TRY_NO_THROW(cusolverSpCreate(&cusolver_res));
            RAFT_CUSOLVER_TRY_NO_THROW(cusolverSpSetStream(cusolver_res, stream));
        }

        void* get_resource() override
        {
            return &cusolver_res;
        }

        ~cusolver_sp_resource() override
        {
            RAFT_CUSOLVER_TRY_NO_THROW(cusolverSpDestroy(cusolver_res));
        }

    private:
        cusolverSpHandle_t cusolver_res;
    };

    /**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
    class cusolver_sp_resource_factory : public resource_factory
    {
    public:
        cusolver_sp_resource_factory(rmm::cuda_stream_view stream)
            : stream_(stream)
        {
        }
        resource_type get_resource_type() override
        {
            return resource_type::CUSOLVER_SP_HANDLE;
        }
        resource* make_resource() override
        {
            return new cusolver_sp_resource(stream_);
        }

    private:
        rmm::cuda_stream_view stream_;
    };

    /**
 * @defgroup resource_cusolver_sp cuSolver SP handle resource functions
 * @{
 */

    /**
 * Load a cusolverSpres_t from raft res if it exists, otherwise
 * add it and return it.
 * @param[in] res the raft resources object
 * @return cusolver sp handle
 */
    inline cusolverSpHandle_t get_cusolver_sp_handle(resources const& res)
    {
        if(!res.has_resource_factory(resource_type::CUSOLVER_SP_HANDLE))
        {
            cudaStream_t stream = get_cuda_stream(res);
            res.add_resource_factory(std::make_shared<cusolver_sp_resource_factory>(stream));
        }
        return *res.get_resource<cusolverSpHandle_t>(resource_type::CUSOLVER_SP_HANDLE);
    };

    /**
 * @}
 */

} // namespace raft::resource
