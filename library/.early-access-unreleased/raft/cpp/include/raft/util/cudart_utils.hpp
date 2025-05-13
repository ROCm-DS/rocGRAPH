// Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <raft/core/error.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <rmm/cuda_stream_view.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/amd_warp_primitives.h>
#include <raft/cuda_runtime.h>
#include <raft/util/warp_primitives.cuh>

#include <hip/hip_fp16.h>
#include <rocprim/rocprim.hpp>
#else
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#endif

#include <execinfo.h>

#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>

namespace raft
{

    /** Helper method to get to know warp size in device code */
    __device__ constexpr inline int warp_size()
    {
#ifdef __HIP_PLATFORM_AMD__
        return hip_warp_primitives::WAVEFRONT_SIZE;
#else
        return 32;
#endif
    }

    __host__ inline int host_warp_size(int device_id)
    {
#ifdef __HIP_PLATFORM_AMD__
        unsigned int warp_size{};
        RAFT_EXPECTS(rocprim::host_warp_size(device_id, warp_size) == hipSuccess,
                     "Failed to query device(ID=%d) for warp size",
                     device_id);
        return warp_size;
#else
        return 32;
#endif
    }

    __host__ inline int host_warp_size(cudaStream_t stream)
    {
#ifdef __HIP_PLATFORM_AMD__
        unsigned int warp_size{};
        RAFT_EXPECTS(rocprim::host_warp_size(stream, warp_size) == hipSuccess,
                     "Failed to query device for warp size");
        return warp_size;
#else
        return 32;
#endif
    }

    __host__ __device__ constexpr inline bitmask_type warp_full_mask()
    {
        return LANE_MASK_ALL;
    }

    /**
 * @brief A kernel grid configuration construction gadget for simple one-dimensional mapping
 * elements to threads.
 */
    class grid_1d_thread_t
    {
    public:
        int const block_size{0};
        int const num_blocks{0};

        /**
   * @param overall_num_elements The number of elements the kernel needs to handle/process
   * @param num_threads_per_block The grid block size, determined according to the kernel's
   * specific features (amount of shared memory necessary, SM functional units use pattern etc.);
   * this can't be determined generically/automatically (as opposed to the number of blocks)
   * @param max_num_blocks_1d maximum number of blocks in 1d grid
   * @param elements_per_thread Typically, a single kernel thread processes more than a single
   * @param device_id Device ID, used to query the specified device for the corresponding hardware
   * warp size. element; this affects the number of threads the grid must contain
   */
        grid_1d_thread_t(size_t overall_num_elements,
                         size_t num_threads_per_block,
                         size_t max_num_blocks_1d,
                         size_t elements_per_thread = 1,
                         int    device_id           = 0)
            : block_size(num_threads_per_block)
            , num_blocks(std::min(
                  (overall_num_elements + (elements_per_thread * num_threads_per_block) - 1)
                      / (elements_per_thread * num_threads_per_block),
                  max_num_blocks_1d))
        {
            RAFT_EXPECTS(overall_num_elements > 0, "overall_num_elements must be > 0");
            RAFT_EXPECTS(num_threads_per_block / raft::host_warp_size(device_id) > 0,
                         "num_threads_per_block / warp_size() must be > 0");
            RAFT_EXPECTS(elements_per_thread > 0, "elements_per_thread must be > 0");
        }
    };

    /**
 * @brief A kernel grid configuration construction gadget for simple one-dimensional mapping
 * elements to warps.
 */
    class grid_1d_warp_t
    {
    public:
        int const block_size{0};
        int const num_blocks{0};

        /**
   * @param overall_num_elements The number of elements the kernel needs to handle/process
   * @param num_threads_per_block The grid block size, determined according to the kernel's
   * specific features (amount of shared memory necessary, SM functional units use pattern etc.);
   * this can't be determined generically/automatically (as opposed to the number of blocks)
   * @param max_num_blocks_1d maximum number of blocks in 1d grid
   * @param device_id Device ID, used to query the specified device for the corresponding hardware
   * warp size.
   */
        grid_1d_warp_t(size_t overall_num_elements,
                       size_t num_threads_per_block,
                       size_t max_num_blocks_1d,
                       int    device_id)
            : block_size(num_threads_per_block)
            , num_blocks(std::min((overall_num_elements
                                   + (num_threads_per_block / raft::host_warp_size(device_id)) - 1)
                                      / (num_threads_per_block / raft::host_warp_size(device_id)),
                                  max_num_blocks_1d))
        {
            RAFT_EXPECTS(overall_num_elements > 0, "overall_num_elements must be > 0");
            RAFT_EXPECTS(num_threads_per_block / raft::host_warp_size(device_id) > 0,
                         "num_threads_per_block / warp_size() must be > 0");
        }
    };

    /**
 * @brief A kernel grid configuration construction gadget for simple one-dimensional mapping
 * elements to blocks.
 */
    class grid_1d_block_t
    {
    public:
        int const block_size{0};
        int const num_blocks{0};

        /**
   * @param overall_num_elements The number of elements the kernel needs to handle/process
   * @param num_threads_per_block The grid block size, determined according to the kernel's
   * specific features (amount of shared memory necessary, SM functional units use pattern etc.);
   * this can't be determined generically/automatically (as opposed to the number of blocks)
   * @param max_num_blocks_1d maximum number of blocks in 1d grid
   * @param device_id Device ID, used to query the specified device for the corresponding hardware
   * warp size.
   */
        grid_1d_block_t(size_t overall_num_elements,
                        size_t num_threads_per_block,
                        size_t max_num_blocks_1d,
                        int    device_id)
            : block_size(num_threads_per_block)
            , num_blocks(std::min(overall_num_elements, max_num_blocks_1d))
        {
            RAFT_EXPECTS(overall_num_elements > 0, "overall_num_elements must be > 0");
            RAFT_EXPECTS(num_threads_per_block / raft::host_warp_size(device_id) > 0,
                         "num_threads_per_block / warp_size() must be > 0");
        }
    };

    /**
 * @brief Generic copy method for all kinds of transfers
 * @tparam Type data type
 * @param dst destination pointer
 * @param src source pointer
 * @param len length of the src/dst buffers in terms of number of elements
 * @param stream cuda stream
 */
    template <typename Type>
    void copy(Type* dst, const Type* src, size_t len, rmm::cuda_stream_view stream)
    {
        RAFT_CUDA_TRY(cudaMemcpyAsync(dst, src, len * sizeof(Type), cudaMemcpyDefault, stream));
    }

    /**
 * @defgroup Copy Copy methods
 * These are here along with the generic 'copy' method in order to improve
 * code readability using explicitly specified function names
 * @{
 */
    /** performs a host to device copy */
    template <typename Type>
    void update_device(Type* d_ptr, const Type* h_ptr, size_t len, rmm::cuda_stream_view stream)
    {
        copy(d_ptr, h_ptr, len, stream);
    }

    /** performs a device to host copy */
    template <typename Type>
    void update_host(Type* h_ptr, const Type* d_ptr, size_t len, rmm::cuda_stream_view stream)
    {
        copy(h_ptr, d_ptr, len, stream);
    }

    template <typename Type>
    void copy_async(Type* d_ptr1, const Type* d_ptr2, size_t len, rmm::cuda_stream_view stream)
    {
        RAFT_CUDA_TRY(
            cudaMemcpyAsync(d_ptr1, d_ptr2, len * sizeof(Type), cudaMemcpyDeviceToDevice, stream));
    }
    /** @} */

    /**
 * @defgroup Debug Utils for debugging host/device buffers
 * @{
 */
    template <class T, class OutStream>
    void print_host_vector(const char* variable_name,
                           const T*    host_mem,
                           size_t      componentsCount,
                           OutStream&  out)
    {
        out << variable_name << "=[";
        for(size_t i = 0; i < componentsCount; ++i)
        {
            if(i != 0)
                out << ",";
            out << host_mem[i];
        }
        out << "];" << std::endl;
    }

    template <class T, class OutStream>
    void print_device_vector(const char* variable_name,
                             const T*    devMem,
                             size_t      componentsCount,
                             OutStream&  out)
    {
        auto host_mem = std::make_unique<T[]>(componentsCount);
        RAFT_CUDA_TRY(cudaMemcpy(
            host_mem.get(), devMem, componentsCount * sizeof(T), cudaMemcpyDeviceToHost));
        print_host_vector(variable_name, host_mem.get(), componentsCount, out);
    }

    /**
 * @brief Print an array given a device or a host pointer.
 *
 * @param[in] variable_name
 * @param[in] ptr any pointer (device/host/managed, etc)
 * @param[in] componentsCount array length
 * @param out the output stream
 */
    template <class T, class OutStream>
    void print_vector(const char* variable_name,
                      const T*    ptr,
                      size_t      componentsCount,
                      OutStream&  out)
    {
        cudaPointerAttributes attr;
        RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, ptr));
        if(attr.hostPointer != nullptr)
        {
            print_host_vector(
                variable_name, reinterpret_cast<T*>(attr.hostPointer), componentsCount, out);
        }
        else if(attr.type == cudaMemoryTypeUnregistered)
        {
            print_host_vector(variable_name, ptr, componentsCount, out);
        }
        else
        {
            print_device_vector(variable_name, ptr, componentsCount, out);
        }
    }
    /** @} */

    /**
 * Returns the id of the device for which the pointer is located
 * @param p pointer to check
 * @return id of device for which pointer is located, otherwise -1.
 */
    template <typename T>
    int get_device_for_address(const T* p)
    {
        if(!p)
        {
            return -1;
        }

        cudaPointerAttributes att;
        cudaError_t           err = cudaPointerGetAttributes(&att, p);
        if(err == cudaErrorInvalidValue)
        {
            // Make sure the current thread error status has been reset
            err = cudaGetLastError();
            return -1;
        }

        // memoryType is deprecated for CUDA 10.0+
        if(att.type == cudaMemoryTypeDevice)
        {
            return att.device;
        }
        else
        {
            return -1;
        }
    }

    /** helper method to get max usable shared mem per block parameter */
    inline int getSharedMemPerBlock()
    {
        int devId;
        RAFT_CUDA_TRY(cudaGetDevice(&devId));
        int smemPerBlk;
        RAFT_CUDA_TRY(
            cudaDeviceGetAttribute(&smemPerBlk, cudaDevAttrMaxSharedMemoryPerBlock, devId));
        return smemPerBlk;
    }

    /** helper method to get multi-processor count parameter */
    inline int getMultiProcessorCount()
    {
        int devId;
        RAFT_CUDA_TRY(cudaGetDevice(&devId));
        int mpCount;
        RAFT_CUDA_TRY(cudaDeviceGetAttribute(&mpCount, cudaDevAttrMultiProcessorCount, devId));
        return mpCount;
    }

    /** helper method to get major minor compute capability version */
    inline std::pair<int, int> getComputeCapability()
    {
        int devId;
        RAFT_CUDA_TRY(cudaGetDevice(&devId));
        int majorVer, minorVer;
        RAFT_CUDA_TRY(cudaDeviceGetAttribute(&majorVer, cudaDevAttrComputeCapabilityMajor, devId));
        RAFT_CUDA_TRY(cudaDeviceGetAttribute(&minorVer, cudaDevAttrComputeCapabilityMinor, devId));

        return std::make_pair(majorVer, minorVer);
    }

    /** helper method to convert an array on device to a string on host */
    template <typename T>
    std::string
        arr2Str(const T* arr, int size, std::string name, cudaStream_t stream, int width = 4)
    {
        std::stringstream ss;

        T* arr_h = (T*)malloc(size * sizeof(T));
        update_host(arr_h, arr, size, stream);
        RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

        ss << name << " = [ ";
        for(int i = 0; i < size; i++)
        {
            typedef typename std::
                conditional_t<std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>, int, T>
                    CastT;

            auto val = static_cast<CastT>(arr_h[i]);
            ss << std::setw(width) << val;

            if(i < size - 1)
                ss << ", ";
        }
        ss << " ]" << std::endl;

        free(arr_h);

        return ss.str();
    }

    /** this seems to be unused, but may be useful in the future */
    template <typename T>
    void ASSERT_DEVICE_MEM(T* ptr, std::string name)
    {
        cudaPointerAttributes s_att;
        cudaError_t           s_err = cudaPointerGetAttributes(&s_att, ptr);

        if(s_err != 0 || s_att.device == -1)
            std::cout << "Invalid device pointer encountered in " << name
                      << ". device=" << s_att.device << ", err=" << s_err << std::endl;
    }

    inline uint32_t curTimeMillis()
    {
        auto now      = std::chrono::high_resolution_clock::now();
        auto duration = now.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    }

    /** Helper function to calculate need memory for allocate to store dense matrix.
 * @param rows number of rows in matrix
 * @param columns number of columns in matrix
 * @return need number of items to allocate via allocate()
 * @sa allocate()
 */
    inline size_t allocLengthForMatrix(size_t rows, size_t columns)
    {
        return rows * columns;
    }

    /** Helper function to check alignment of pointer.
 * @param ptr the pointer to check
 * @param alignment to be checked for
 * @return true if address in bytes is a multiple of alignment
 */
    template <typename Type>
    bool is_aligned(Type* ptr, size_t alignment)
    {
        return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
    }

    /** calculate greatest common divisor of two numbers
 * @a integer
 * @b integer
 * @ return gcd of a and b
 */
    template <typename IntType>
    constexpr IntType gcd(IntType a, IntType b)
    {
        while(b != 0)
        {
            IntType tmp = b;
            b           = a % b;
            a           = tmp;
        }
        return a;
    }

    template <typename T>
    constexpr T lower_bound()
    {
        if constexpr(std::numeric_limits<T>::has_infinity && std::numeric_limits<T>::is_signed)
        {
            return -std::numeric_limits<T>::infinity();
        }
        return std::numeric_limits<T>::lowest();
    }

    template <typename T>
    constexpr T upper_bound()
    {
        if constexpr(std::numeric_limits<T>::has_infinity)
        {
            return std::numeric_limits<T>::infinity();
        }
        return std::numeric_limits<T>::max();
    }

    namespace
    { // NOLINT
        /**
 * This is a hack to allow constexpr definition of `half` constants.
 *
 * Neither union-based nor reinterpret_cast-based type punning is possible within
 * constexpr; at the same time, all non-default constructors of `half` data type are not constexpr
 * as well.
 *
 * Based on the implementation details in `cuda_fp16.hpp`, we define here a new constructor for
 * `half` data type, that is a proper constexpr.
 *
 * When we switch to C++20, perhaps we can use `bit_cast` for the same purpose.
 */
        struct __half_constexpr : __half
        { // NOLINT
            constexpr explicit inline __half_constexpr(uint16_t u)
                : __half()
            {
                __x = u;
            }
        };
    } // namespace

    template <>
    constexpr inline auto lower_bound<half>() -> half
    {
        return static_cast<half>(__half_constexpr{0xfc00u});
    }

    template <>
    constexpr inline auto upper_bound<half>() -> half
    {
        return static_cast<half>(__half_constexpr{0x7c00u});
    }

} // namespace raft
