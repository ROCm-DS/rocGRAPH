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
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/detail/stream_sync_event.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#include <rmm/cuda_stream_pool.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace raft::resource
{

    class cuda_stream_pool_resource : public resource
    {
    public:
        cuda_stream_pool_resource(std::shared_ptr<rmm::cuda_stream_pool> stream_pool)
            : stream_pool_(stream_pool)
        {
        }

        ~cuda_stream_pool_resource() override {}
        void* get_resource() override
        {
            return &stream_pool_;
        }

    private:
        std::shared_ptr<rmm::cuda_stream_pool> stream_pool_{nullptr};
    };

    /**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
    class cuda_stream_pool_resource_factory : public resource_factory
    {
    public:
        cuda_stream_pool_resource_factory(std::shared_ptr<rmm::cuda_stream_pool> stream_pool
                                          = {nullptr})
            : stream_pool_(stream_pool)
        {
        }

        resource_type get_resource_type() override
        {
            return resource_type::CUDA_STREAM_POOL;
        }
        resource* make_resource() override
        {
            return new cuda_stream_pool_resource(stream_pool_);
        }

    private:
        std::shared_ptr<rmm::cuda_stream_pool> stream_pool_{nullptr};
    };

    inline bool is_stream_pool_initialized(const resources& res)
    {
        return *res.get_resource<std::shared_ptr<rmm::cuda_stream_pool>>(
                   resource_type::CUDA_STREAM_POOL)
               != nullptr;
    }

    /**
 * @defgroup resource_stream_pool CUDA Stream pool resource functions
 * @{
 */

    /**
 * Load a cuda_stream_pool, and create a new one if it doesn't already exist
 * @param res raft res object for managing resources
 * @return
 */
    inline const rmm::cuda_stream_pool& get_cuda_stream_pool(const resources& res)
    {
        if(!res.has_resource_factory(resource_type::CUDA_STREAM_POOL))
        {
            res.add_resource_factory(std::make_shared<cuda_stream_pool_resource_factory>());
        }
        return *(*res.get_resource<std::shared_ptr<rmm::cuda_stream_pool>>(
            resource_type::CUDA_STREAM_POOL));
    };

    /**
 * Explicitly set a stream pool on the current res. Note that this will overwrite
 * an existing stream pool on the res.
 * @param res
 * @param stream_pool
 */
    inline void set_cuda_stream_pool(const resources&                       res,
                                     std::shared_ptr<rmm::cuda_stream_pool> stream_pool)
    {
        res.add_resource_factory(std::make_shared<cuda_stream_pool_resource_factory>(stream_pool));
    };

    inline std::size_t get_stream_pool_size(const resources& res)
    {
        return is_stream_pool_initialized(res) ? get_cuda_stream_pool(res).get_pool_size() : 0;
    }

    /**
 * @brief return stream from pool
 */
    inline rmm::cuda_stream_view get_stream_from_stream_pool(const resources& res)
    {
        RAFT_EXPECTS(is_stream_pool_initialized(res),
                     "ERROR: rmm::cuda_stream_pool was not initialized");
        return get_cuda_stream_pool(res).get_stream();
    }

    /**
 * @brief return stream from pool at index
 */
    inline rmm::cuda_stream_view get_stream_from_stream_pool(const resources& res,
                                                             std::size_t      stream_idx)
    {
        RAFT_EXPECTS(is_stream_pool_initialized(res),
                     "ERROR: rmm::cuda_stream_pool was not initialized");
        return get_cuda_stream_pool(res).get_stream(stream_idx);
    }

    /**
 * @brief return stream from pool if size > 0, else main stream on res
 */
    inline rmm::cuda_stream_view get_next_usable_stream(const resources& res)
    {
        return is_stream_pool_initialized(res) ? get_stream_from_stream_pool(res)
                                               : get_cuda_stream(res);
    }

    /**
 * @brief return stream from pool at index if size > 0, else main stream on res
 *
 * @param[in] res the raft resources object
 * @param[in] stream_idx the required index of the stream in the stream pool if available
 */
    inline rmm::cuda_stream_view get_next_usable_stream(const resources& res,
                                                        std::size_t      stream_idx)
    {
        return is_stream_pool_initialized(res) ? get_stream_from_stream_pool(res, stream_idx)
                                               : get_cuda_stream(res);
    }

    /**
 * @brief synchronize the stream pool on the res
 *
 * @param[in] res the raft resources object
 */
    inline void sync_stream_pool(const resources& res)
    {
        for(std::size_t i = 0; i < get_stream_pool_size(res); i++)
        {
            sync_stream(res, get_cuda_stream_pool(res).get_stream(i));
        }
    }

    /**
 * @brief synchronize subset of stream pool
 *
 * @param[in] res the raft resources object
 * @param[in] stream_indices the indices of the streams in the stream pool to synchronize
 */
    inline void sync_stream_pool(const resources&               res,
                                 const std::vector<std::size_t> stream_indices)
    {
        RAFT_EXPECTS(is_stream_pool_initialized(res),
                     "ERROR: rmm::cuda_stream_pool was not initialized");
        for(const auto& stream_index : stream_indices)
        {
            sync_stream(res, get_cuda_stream_pool(res).get_stream(stream_index));
        }
    }

    /**
 * @brief ask stream pool to wait on last event in main stream
 *
 * @param[in] res the raft resources object
 */
    inline void wait_stream_pool_on_stream(const resources& res)
    {
        if(!res.has_resource_factory(resource_type::CUDA_STREAM_POOL))
        {
            res.add_resource_factory(std::make_shared<cuda_stream_pool_resource_factory>());
        }

        cudaEvent_t event = detail::get_cuda_stream_sync_event(res);
        RAFT_CUDA_TRY(cudaEventRecord(event, get_cuda_stream(res)));
        for(std::size_t i = 0; i < get_stream_pool_size(res); i++)
        {
            RAFT_CUDA_TRY(cudaStreamWaitEvent(get_cuda_stream_pool(res).get_stream(i), event, 0));
        }
    }

    /**
 * @}
 */

} // namespace raft::resource
