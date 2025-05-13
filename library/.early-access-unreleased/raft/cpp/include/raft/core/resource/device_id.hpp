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

    class device_id_resource : public resource
    {
    public:
        device_id_resource()
            : dev_id_([]() -> int {
                int cur_dev = -1;
                RAFT_CUDA_TRY_NO_THROW(cudaGetDevice(&cur_dev));
                return cur_dev;
            }())
        {
        }
        void* get_resource() override
        {
            return &dev_id_;
        }

        ~device_id_resource() override {}

    private:
        int dev_id_;
    };

    /**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
    class device_id_resource_factory : public resource_factory
    {
    public:
        resource_type get_resource_type() override
        {
            return resource_type::DEVICE_ID;
        }
        resource* make_resource() override
        {
            return new device_id_resource();
        }
    };

    /**
 * @defgroup resource_device_id Device ID resource functions
 * @{
 */

    /**
 * Load a device id from a res (and populate it on the res if needed).
 * @param res raft res object for managing resources
 * @return device id
 */
    inline int get_device_id(resources const& res)
    {
        if(!res.has_resource_factory(resource_type::DEVICE_ID))
        {
            res.add_resource_factory(std::make_shared<device_id_resource_factory>());
        }
        return *res.get_resource<int>(resource_type::DEVICE_ID);
    };

    /**
 * @}
 */
} // namespace raft::resource
