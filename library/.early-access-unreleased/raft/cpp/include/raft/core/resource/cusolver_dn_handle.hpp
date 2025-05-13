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

#include "cuda_stream.hpp"

#include <raft/core/cusolver_macros.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#include <rmm/cuda_stream_view.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cusolver.h>
#else
#include <cusolverDn.h>
#endif

namespace raft::resource
{

    /**
 *
 */
    class cusolver_dn_resource : public resource
    {
    public:
        cusolver_dn_resource(rmm::cuda_stream_view stream)
        {
            RAFT_CUSOLVER_TRY_NO_THROW(cusolverDnCreate(&cusolver_res));
            RAFT_CUSOLVER_TRY_NO_THROW(cusolverDnSetStream(cusolver_res, stream));
        }

        void* get_resource() override
        {
            return &cusolver_res;
        }

        ~cusolver_dn_resource() override
        {
            RAFT_CUSOLVER_TRY_NO_THROW(cusolverDnDestroy(cusolver_res));
        }

    private:
        cusolverDnHandle_t cusolver_res;
    };

    /**
 * @defgroup resource_cusolver_dn cuSolver DN handle resource functions
 * @{
 */

    /**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
    class cusolver_dn_resource_factory : public resource_factory
    {
    public:
        cusolver_dn_resource_factory(rmm::cuda_stream_view stream)
            : stream_(stream)
        {
        }
        resource_type get_resource_type() override
        {
            return resource_type::CUSOLVER_DN_HANDLE;
        }
        resource* make_resource() override
        {
            return new cusolver_dn_resource(stream_);
        }

    private:
        rmm::cuda_stream_view stream_;
    };

    /**
 * Load a cusolverSpres_t from raft res if it exists, otherwise
 * add it and return it.
 * @param[in] res the raft resources object
 * @return cusolver dn handle
 */
    inline cusolverDnHandle_t get_cusolver_dn_handle(resources const& res)
    {
        if(!res.has_resource_factory(resource_type::CUSOLVER_DN_HANDLE))
        {
            cudaStream_t stream = get_cuda_stream(res);
            res.add_resource_factory(std::make_shared<cusolver_dn_resource_factory>(stream));
        }
        return *res.get_resource<cusolverDnHandle_t>(resource_type::CUSOLVER_DN_HANDLE);
    };

    /**
 * @}
 */

} // namespace raft::resource
