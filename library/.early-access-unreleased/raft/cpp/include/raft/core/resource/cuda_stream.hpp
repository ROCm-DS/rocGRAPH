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

#include <raft/core/interruptible.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream_view.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace raft::resource
{
    class cuda_stream_resource : public resource
    {
    public:
        cuda_stream_resource(rmm::cuda_stream_view stream_view = rmm::cuda_stream_per_thread)
            : stream(stream_view)
        {
        }
        void* get_resource() override
        {
            return &stream;
        }

        ~cuda_stream_resource() override {}

    private:
        rmm::cuda_stream_view stream;
    };

    /**
 * Factory that knows how to construct a specific raft::resource to populate
 * the resources instance.
 */
    class cuda_stream_resource_factory : public resource_factory
    {
    public:
        cuda_stream_resource_factory(rmm::cuda_stream_view stream_view
                                     = rmm::cuda_stream_per_thread)
            : stream(stream_view)
        {
        }
        resource_type get_resource_type() override
        {
            return resource_type::CUDA_STREAM_VIEW;
        }
        resource* make_resource() override
        {
            return new cuda_stream_resource(stream);
        }

    private:
        rmm::cuda_stream_view stream;
    };

    /**
 * @defgroup resource_cuda_stream CUDA stream resource functions
 * @{
 */
    /**
 * Load a rmm::cuda_stream_view from a resources instance (and populate it on the res
 * if needed).
 * @param res raft res object for managing resources
 * @return
 */
    inline rmm::cuda_stream_view get_cuda_stream(resources const& res)
    {
        if(!res.has_resource_factory(resource_type::CUDA_STREAM_VIEW))
        {
            res.add_resource_factory(std::make_shared<cuda_stream_resource_factory>());
        }
        return *res.get_resource<rmm::cuda_stream_view>(resource_type::CUDA_STREAM_VIEW);
    };

    /**
 * Load a rmm::cuda_stream_view from a resources instance (and populate it on the res
 * if needed).
 * @param[in] res raft resources object for managing resources
 * @param[in] stream_view cuda stream view
 */
    inline void set_cuda_stream(resources const& res, rmm::cuda_stream_view stream_view)
    {
        res.add_resource_factory(std::make_shared<cuda_stream_resource_factory>(stream_view));
    };

    /**
 * @brief synchronize a specific stream
 *
 * @param[in] res the raft resources object
 * @param[in] stream stream to synchronize
 */
    inline void sync_stream(const resources& res, rmm::cuda_stream_view stream)
    {
        interruptible::synchronize(stream);
    }

    /**
 * @brief synchronize main stream on the resources instance
 */
    inline void sync_stream(const resources& res)
    {
        sync_stream(res, get_cuda_stream(res));
    }

    /**
 * @}
 */

} // namespace raft::resource
