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

#include <raft/core/cusparse_macros.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cusparse.h>
#else
#include <cusparse_v2.h>
#endif

namespace raft::resource
{
    class cusparse_resource : public resource
    {
    public:
        cusparse_resource(rmm::cuda_stream_view stream)
        {
            RAFT_CUSPARSE_TRY_NO_THROW(cusparseCreate(&cusparse_res));
            RAFT_CUSPARSE_TRY_NO_THROW(cusparseSetStream(cusparse_res, stream));
        }

        ~cusparse_resource()
        {
            RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroy(cusparse_res));
        }
        void* get_resource() override
        {
            return &cusparse_res;
        }

    private:
        cusparseHandle_t cusparse_res;
    };

    /**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
    class cusparse_resource_factory : public resource_factory
    {
    public:
        cusparse_resource_factory(rmm::cuda_stream_view stream)
            : stream_(stream)
        {
        }
        resource_type get_resource_type() override
        {
            return resource_type::CUSPARSE_HANDLE;
        }
        resource* make_resource() override
        {
            return new cusparse_resource(stream_);
        }

    private:
        rmm::cuda_stream_view stream_;
    };

    /**
 * @defgroup resource_cusparse cuSparse handle resource functions
 * @{
 */

    /**
 * Load a cusparseres_t from raft res if it exists, otherwise
 * add it and return it.
 * @param[in] res the raft resources object
 * @return cusparse handle
 */
    inline cusparseHandle_t get_cusparse_handle(resources const& res)
    {
        if(!res.has_resource_factory(resource_type::CUSPARSE_HANDLE))
        {
            rmm::cuda_stream_view stream = get_cuda_stream(res);
            res.add_resource_factory(std::make_shared<cusparse_resource_factory>(stream));
        }
        return *res.get_resource<cusparseHandle_t>(resource_type::CUSPARSE_HANDLE);
    };

    /**
 * @}
 */

} // namespace raft::resource
