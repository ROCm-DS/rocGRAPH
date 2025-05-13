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

#include <raft/core/resource/device_id.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace raft::resource
{

    class device_properties_resource : public resource
    {
    public:
        device_properties_resource(int dev_id)
        {
            RAFT_CUDA_TRY_NO_THROW(cudaGetDeviceProperties(&prop_, dev_id));
        }
        void* get_resource() override
        {
            return &prop_;
        }

        ~device_properties_resource() override {}

    private:
        cudaDeviceProp prop_;
    };

    /**
 * @defgroup resource_device_props Device properties resource functions
 * @{
 */

    /**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
    class device_properties_resource_factory : public resource_factory
    {
    public:
        device_properties_resource_factory(int dev_id)
            : dev_id_(dev_id)
        {
        }
        resource_type get_resource_type() override
        {
            return resource_type::DEVICE_PROPERTIES;
        }
        resource* make_resource() override
        {
            return new device_properties_resource(dev_id_);
        }

    private:
        int dev_id_;
    };

    /**
 * Load a cudaDeviceProp from a res (and populate it on the res if needed).
 * @param res raft res object for managing resources
 * @return populated cuda device properties instance
 */
    inline cudaDeviceProp& get_device_properties(resources const& res)
    {
        if(!res.has_resource_factory(resource_type::DEVICE_PROPERTIES))
        {
            int dev_id = get_device_id(res);
            res.add_resource_factory(std::make_shared<device_properties_resource_factory>(dev_id));
        }
        return *res.get_resource<cudaDeviceProp>(resource_type::DEVICE_PROPERTIES);
    };

    /**
 * @}
 */
} // namespace raft::resource
