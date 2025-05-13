/*! \file */

/*
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "control.h"
#include "enum_utils.hpp"
#include "handle.h"
namespace rocgraph
{
// Return the leftmost significant bit position
#if defined(rocgraph_ILP64)
    static inline rocgraph_int clz(rocgraph_int n)
    {
        // __builtin_clzll is undefined for n == 0
        if(n == 0)
        {
            return 0;
        }
        return 64 - __builtin_clzll(n);
    }
#else
    static inline rocgraph_int clz(rocgraph_int n)
    {
        // __builtin_clz is undefined for n == 0
        if(n == 0)
        {
            return 0;
        }
        return 32 - __builtin_clz(n);
    }
#endif

#if 0
    // Return one on the device
    static inline void one(const rocgraph_handle handle, float** one)
    {
        *one = handle->sone;
    }

    static inline void one(const rocgraph_handle handle, double** one)
    {
        *one = handle->done;
    }
#endif
    template <typename T>
    ROCGRAPH_KERNEL(1)
    void assign_kernel(T* dest, T value)
    {
        *dest = value;
    }

    // Set a single value on the device from the host asynchronously.
    template <typename T>
    static inline hipError_t assign_async(T* dest, T value, hipStream_t stream)
    {
        // Use a kernel instead of memcpy, because memcpy is synchronous if the source is not in
        // pinned memory.
        // Memset lacks a 64bit option, but would involve a similar implicit kernel anyways.

        if(false == rocgraph_debug_variables.get_debug_kernel_launch())
        {
            hipLaunchKernelGGL(rocgraph::assign_kernel, dim3(1), dim3(1), 0, stream, dest, value);
            return hipSuccess;
        }
        else
        {
            {
                const hipError_t err = hipGetLastError();
                if(err != hipSuccess)
                {
                    std::stringstream s;
                    s << "prior to hipLaunchKernelGGL"
                      << ", hip error detected: code '" << err << "', name '"
                      << hipGetErrorName(err) << "', description '" << hipGetErrorString(err)
                      << "'";
                    ROCGRAPH_ERROR_MESSAGE(rocgraph::get_rocgraph_status_for_hip_status(err),
                                           s.str().c_str());
                    return err;
                }
            }
            hipLaunchKernelGGL(rocgraph::assign_kernel, dim3(1), dim3(1), 0, stream, dest, value);
            {
                const hipError_t err = hipGetLastError();
                if(err != hipSuccess)
                {
                    std::stringstream s;
                    s << "hip error detected: code '" << err << "', name '" << hipGetErrorName(err)
                      << "', description '" << hipGetErrorString(err) << "'";
                    ROCGRAPH_ERROR_MESSAGE(rocgraph::get_rocgraph_status_for_hip_status(err),
                                           s.str().c_str());
                    return err;
                }
            }
            return hipSuccess;
        }
    }

    // For host scalars
    template <typename T>
    __forceinline__ __device__ __host__ T load_scalar_device_host(T x)
    {
        return x;
    }

    // For device scalars
    template <typename T>
    __forceinline__ __device__ __host__ T load_scalar_device_host(const T* xp)
    {
        return *xp;
    }

    // For host scalars
    template <typename T>
    __forceinline__ __device__ __host__ T zero_scalar_device_host(T x)
    {
        return static_cast<T>(0);
    }

    // For device scalars
    template <typename T>
    __forceinline__ __device__ __host__ T zero_scalar_device_host(const T* xp)
    {
        return static_cast<T>(0);
    }

    template <typename T>
    struct floating_traits
    {
        using data_t = T;
    };

    template <typename T>
    using floating_data_t = typename floating_traits<T>::data_t;

    template <typename T>
    rocgraph_indextype get_indextype();

    template <>
    inline rocgraph_indextype get_indextype<int32_t>()
    {
        return rocgraph_indextype_i32;
    }

    template <>
    inline rocgraph_indextype get_indextype<uint16_t>()
    {
        return rocgraph_indextype_u16;
    }

    template <>
    inline rocgraph_indextype get_indextype<int64_t>()
    {
        return rocgraph_indextype_i64;
    }

    template <typename T>
    rocgraph_datatype get_datatype();

    template <>
    inline rocgraph_datatype get_datatype<float>()
    {
        return rocgraph_datatype_f32_r;
    }

    template <>
    inline rocgraph_datatype get_datatype<double>()
    {
        return rocgraph_datatype_f64_r;
    }

    inline size_t indextype_sizeof(rocgraph_indextype that)
    {
        switch(that)
        {

        case rocgraph_indextype_i32:
        {
            return sizeof(int32_t);
        }
        case rocgraph_indextype_i64:
        {
            return sizeof(int64_t);
        }
        case rocgraph_indextype_u16:
        {
            return sizeof(uint16_t);
        }
        }
    }

    inline size_t datatype_sizeof(rocgraph_datatype that)
    {
        switch(that)
        {
        case rocgraph_datatype_i32_r:
            return sizeof(int32_t);

        case rocgraph_datatype_u32_r:
        {
            return sizeof(uint32_t);
        }

        case rocgraph_datatype_i8_r:
        {
            return sizeof(int8_t);
        }

        case rocgraph_datatype_u8_r:
        {
            return sizeof(uint8_t);
        }
        case rocgraph_datatype_f32_r:
        {
            return sizeof(float);
        }

        case rocgraph_datatype_f64_r:
        {
            return sizeof(double);
        }
        }
    }

#include "memstat.h"

    inline rocgraph_status calculate_nnz(
        int64_t m, rocgraph_indextype indextype, const void* ptr, int64_t* nnz, hipStream_t stream)
    {
        if(m == 0)
        {
            nnz[0] = 0;
            return rocgraph_status_success;
        }
        const char* p
            = reinterpret_cast<const char*>(ptr) + rocgraph::indextype_sizeof(indextype) * m;
        int64_t end, start;
        switch(indextype)
        {
        case rocgraph_indextype_i32:
        {
            int32_t u, v;
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &u, ptr, rocgraph::indextype_sizeof(indextype), hipMemcpyDeviceToHost, stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &v, p, rocgraph::indextype_sizeof(indextype), hipMemcpyDeviceToHost, stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
            start = u;
            end   = v;
            break;
        }
        case rocgraph_indextype_i64:
        {
            int64_t u, v;
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &u, ptr, rocgraph::indextype_sizeof(indextype), hipMemcpyDeviceToHost, stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &v, p, rocgraph::indextype_sizeof(indextype), hipMemcpyDeviceToHost, stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
            start = u;
            end   = v;
            break;
        }
        case rocgraph_indextype_u16:
        {
            uint16_t u, v;
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &u, ptr, rocgraph::indextype_sizeof(indextype), hipMemcpyDeviceToHost, stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &v, p, rocgraph::indextype_sizeof(indextype), hipMemcpyDeviceToHost, stream));
            start = u;
            end   = v;
            break;
        }
        }
        nnz[0] = end - start;
        return rocgraph_status_success;
    }

    template <typename S, typename T>
    inline rocgraph_status internal_convert_scalar(const S s, T& t)
    {
        if(s <= std::numeric_limits<T>::max() && s >= std::numeric_limits<T>::min())
        {
            t = static_cast<T>(s);
            return rocgraph_status_success;
        }
        else
        {
            RETURN_IF_ROCGRAPH_ERROR(rocgraph_status_type_mismatch);
        }
    }
}
