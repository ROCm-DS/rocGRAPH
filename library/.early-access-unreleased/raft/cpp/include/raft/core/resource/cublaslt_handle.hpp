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

#include <raft/core/cublas_macros.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cublasLt.h>
#else
#include <cublasLt.h>
#endif

#include <memory>

namespace raft::resource
{

    class cublaslt_resource : public resource
    {
    public:
        cublaslt_resource()
        {
            RAFT_CUBLAS_TRY(cublasLtCreate(&handle_));
        }
        ~cublaslt_resource() noexcept override
        {
            RAFT_CUBLAS_TRY_NO_THROW(cublasLtDestroy(handle_));
        }
        auto get_resource() -> void* override
        {
            return &handle_;
        }

    private:
        cublasLtHandle_t handle_;
    };

    /** Factory that knows how to construct a specific raft::resource to populate the res_t. */
    class cublaslt_resource_factory : public resource_factory
    {
    public:
        auto get_resource_type() -> resource_type override
        {
            return resource_type::CUBLASLT_HANDLE;
        }
        auto make_resource() -> resource* override
        {
            return new cublaslt_resource();
        }
    };

    /**
 * @defgroup resource_cublaslt cuBLASLt handle resource functions
 * @{
 */

    /**
 * Load a `cublasLtHandle_t` from raft res if it exists, otherwise add it and return it.
 *
 * @param[in] res the raft resources object
 * @return cublasLt handle
 */
    inline auto get_cublaslt_handle(resources const& res) -> cublasLtHandle_t
    {
        if(!res.has_resource_factory(resource_type::CUBLASLT_HANDLE))
        {
            res.add_resource_factory(std::make_shared<cublaslt_resource_factory>());
        }
        auto ret = *res.get_resource<cublasLtHandle_t>(resource_type::CUBLASLT_HANDLE);
        return ret;
    };

    /**
 * @}
 */

} // namespace raft::resource
