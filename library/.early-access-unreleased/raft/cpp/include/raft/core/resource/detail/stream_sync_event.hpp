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

#include <raft/core/resource/cuda_event.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace raft::resource::detail
{

    /**
 * Factory that knows how to construct a specific raft::resource to populate
 * the res_t.
 */
    class cuda_stream_sync_event_resource_factory : public resource_factory
    {
    public:
        resource_type get_resource_type() override
        {
            return resource_type::CUDA_STREAM_SYNC_EVENT;
        }
        resource* make_resource() override
        {
            return new cuda_event_resource();
        }
    };

    /**
 * Load a cudaEvent from a resources instance (and populate it on the resources instance)
 * if needed) for syncing the main cuda stream.
 * @param res raft resources instance for managing resources
 * @return
 */
    inline cudaEvent_t& get_cuda_stream_sync_event(resources const& res)
    {
        if(!res.has_resource_factory(resource_type::CUDA_STREAM_SYNC_EVENT))
        {
            res.add_resource_factory(std::make_shared<cuda_stream_sync_event_resource_factory>());
        }
        return *res.get_resource<cudaEvent_t>(resource_type::CUDA_STREAM_SYNC_EVENT);
    };

} // namespace raft::resource::detail
