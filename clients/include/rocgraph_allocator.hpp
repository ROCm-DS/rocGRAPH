/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef ROCGRAPH_ALLOCATOR_HPP
#define ROCGRAPH_ALLOCATOR_HPP

#include "rocgraph_random.hpp"
#include <hip/hip_runtime_api.h>
#ifdef GOOGLE_TEST
#include "rocgraph_test.hpp"
#endif
#include <cinttypes>
#include <iostream>
#include <locale.h>
struct memory_mode
{
    typedef enum _value
    {
        device = 0,
        host,
        managed
    } value_t;

    static constexpr hipMemcpyKind get_hipMemcpyKind(memory_mode::value_t TARGET,
                                                     memory_mode::value_t SOURCE)
    {
        switch(TARGET)
        {
        case memory_mode::host:
        {
            switch(SOURCE)
            {
            case memory_mode::host:
            {
                return hipMemcpyHostToHost;
            }
            case memory_mode::device:
            {
                return hipMemcpyDeviceToHost;
            }
            case memory_mode::managed:
            {
                return hipMemcpyHostToHost;
            }
            }
        }
        case memory_mode::device:
        {
            switch(SOURCE)
            {
            case memory_mode::host:
            {
                return hipMemcpyHostToDevice;
            }
            case memory_mode::device:
            {
                return hipMemcpyDeviceToDevice;
            }
            case memory_mode::managed:
            {
                return hipMemcpyDeviceToDevice;
            }
            }
        }
        case memory_mode::managed:
        {
            switch(SOURCE)
            {
            case memory_mode::host:
            {
                return hipMemcpyHostToHost;
            }
            case memory_mode::managed:
            {
                return hipMemcpyHostToHost;
            }
            case memory_mode::device:
            {
                return hipMemcpyDeviceToDevice;
            }
            }
        }
        }
        throw(rocgraph_status_invalid_value);
    }
};

#ifndef ROCGRAPH_WITH_MEMSTAT

#define rocgraph_hipMalloc(p_, nbytes_) hipMalloc(p_, nbytes_)

#define rocgraph_hipFree(p_) hipFree(p_)

#define rocgraph_hipHostMalloc(p_, nbytes_) hipHostMalloc(p_, nbytes_)
#define rocgraph_hipHostFree(p_) hipHostFree(p_)

#define rocgraph_hipMallocManaged(p_, nbytes_) hipMallocManaged(p_, nbytes_)
#define rocgraph_hipFreeManaged(p_) hipFree(p_)

#else

template <typename T>
hipError_t rocgraph_hip_free_template(T* mem, const char* tag)
{
    return rocgraph_hip_free(static_cast<void*>(mem), tag);
}

template <typename T>
hipError_t rocgraph_hip_malloc_template(T** mem, size_t nbytes, const char* tag)
{
    void* m;
    auto  error = rocgraph_hip_malloc(&m, nbytes, tag);
    if(error != hipSuccess)
    {
        return error;
    }
    mem[0] = static_cast<T*>(m);
    return hipSuccess;
}

template <typename T>
hipError_t rocgraph_hip_host_free_template(T* mem, const char* tag)
{
    return rocgraph_hip_host_free(static_cast<void*>(mem), tag);
}

template <typename T>
hipError_t rocgraph_hip_host_malloc_template(T** mem, size_t nbytes, const char* tag)
{
    void* m;
    auto  error = rocgraph_hip_host_malloc(&m, nbytes, tag);
    if(error != hipSuccess)
    {
        return error;
    }
    mem[0] = static_cast<T*>(m);
    return hipSuccess;
}
template <typename T>
hipError_t rocgraph_hip_free_managed_template(T* mem, const char* tag)
{
    return rocgraph_hip_free_managed(static_cast<void*>(mem), tag);
}

template <typename T>
hipError_t rocgraph_hip_malloc_managed_template(T** mem, size_t nbytes, const char* tag)
{
    void* m;
    auto  error = rocgraph_hip_malloc_managed(&m, nbytes, tag);
    if(error != hipSuccess)
    {
        return error;
    }
    mem[0] = static_cast<T*>(m);
    return hipSuccess;
}

#define ROCGRAPH_CLIENTS_MEMORY_SOURCE_LINE(line_) #line_
#define ROCGRAPH_CLIENTS_MEMORY_SOURCE_TAG(line_) \
    __FILE__ " " ROCGRAPH_CLIENTS_MEMORY_SOURCE_LINE(line_)

#define rocgraph_hipMalloc(p_, nbytes_) \
    rocgraph_hip_malloc_template(p_, nbytes_, ROCGRAPH_CLIENTS_MEMORY_SOURCE_TAG(__LINE__))

#define rocgraph_hipFree(p_) \
    rocgraph_hip_free_template(p_, ROCGRAPH_CLIENTS_MEMORY_SOURCE_TAG(__LINE__))

#define rocgraph_hipHostMalloc(p_, nbytes_) \
    rocgraph_hip_host_malloc_template(p_, nbytes_, ROCGRAPH_CLIENTS_MEMORY_SOURCE_TAG(__LINE__))

#define rocgraph_hipHostFree(p_) \
    rocgraph_hip_host_free_template(p_, ROCGRAPH_CLIENTS_MEMORY_SOURCE_TAG(__LINE__))

#define rocgraph_hipMallocManaged(p_, nbytes_) \
    rocgraph_hip_malloc_managed_template(p_, nbytes_, ROCGRAPH_CLIENTS_MEMORY_SOURCE_TAG(__LINE__))

#define rocgraph_hipFreeManaged(p_) \
    rocgraph_hip_free_managed_template(p_, ROCGRAPH_CLIENTS_MEMORY_SOURCE_TAG(__LINE__))

#endif

template <memory_mode::value_t MODE, typename T, size_t PAD = 4096, typename U = T>
struct rocgraph_allocator
{

private:
    static size_t compute_nbytes(size_t s)
    {
#ifdef GOOGLE_TEST
        return sizeof(T) * (s + PAD * 3);
#else
        return sizeof(T) * s;
#endif
    }

#ifdef GOOGLE_TEST
    static void init_guards(U* A, size_t N)
    {
        for(size_t i = 0; i < N; ++i)
            A[i] = U(rocgraph_nan_rng());
    }

    static T* off_guards(T* d)
    {
        if(PAD > 0)
        {
            d = (T*)(((U*)d) - 2 * PAD);
        }
        return d;
    }

    static T* install_guards(T* d, size_t size)
    {
        if(d != nullptr)
        {
            if(PAD > 0)
            {
                U guard[PAD];
                init_guards(guard, PAD);

                // Copy guard to device memory before allocated memory
                hipMemcpy(d,
                          guard,
                          sizeof(guard),
                          memory_mode::get_hipMemcpyKind(MODE, memory_mode::host));
                hipMemcpy(d + PAD,
                          guard,
                          sizeof(guard),
                          memory_mode::get_hipMemcpyKind(MODE, memory_mode::host));

                // Point to allocated block
                d += 2 * PAD;

                // Copy guard to device memory after allocated memory
                hipMemcpy(d + size,
                          guard,
                          sizeof(guard),
                          memory_mode::get_hipMemcpyKind(MODE, memory_mode::host));
            }
        }
        return d;
    }
#endif

public:
    static T* malloc(size_t size)
    {
        if(size == 0)
        {
            return (T*)nullptr;
        }
        size_t nbytes = compute_nbytes(size);
        T*     d;
        switch(MODE)
        {
        case memory_mode::host:
        {
            if(rocgraph_hipHostMalloc(&d, nbytes) != hipSuccess)
            {
                fprintf(stderr, "Error allocating %'zu bytes (%zu GB)\n", nbytes, nbytes >> 30);
                d = nullptr;
                throw std::bad_alloc();
            }
            break;
        }
        case memory_mode::device:
        {
            if(rocgraph_hipMalloc(&d, nbytes) != hipSuccess)
            {
                fprintf(stderr, "Error allocating %'zu bytes (%zu GB)\n", nbytes, nbytes >> 30);
                d = nullptr;
                throw std::bad_alloc();
            }
            break;
        }
        case memory_mode::managed:
        {

            if(rocgraph_hipMallocManaged(&d, nbytes) != hipSuccess)
            {
                fprintf(stderr, "Error allocating %'zu bytes (%zu GB)\n", nbytes, nbytes >> 30);
                d = nullptr;
                throw std::bad_alloc();
            }

            break;
        }
        }
#ifdef GOOGLE_TEST
        return install_guards(d, size);
#else
        return d;
#endif
    }

#ifdef GOOGLE_TEST
    static void check_guards(T* d, size_t size)
    {
        if(d != nullptr)
        {
            if(PAD > 0)
            {
                U host[PAD], guard[PAD];
                // Copy device memory after allocated memory to host
                hipMemcpy(guard,
                          ((U*)d) - 2 * PAD,
                          sizeof(guard),
                          memory_mode::get_hipMemcpyKind(memory_mode::host, MODE));

                // Copy device memory after allocated memory to host
                hipMemcpy(host,
                          d + size,
                          sizeof(guard),
                          memory_mode::get_hipMemcpyKind(memory_mode::host, MODE));

                // Make sure no corruption has occurred
                EXPECT_EQ(memcmp(host, guard, sizeof(guard)), 0);

                // Point to guard before allocated memory
                d -= PAD;

                // Copy device memory after allocated memory to host
                hipMemcpy(host,
                          d,
                          sizeof(guard),
                          memory_mode::get_hipMemcpyKind(memory_mode::host, MODE));

                // Make sure no corruption has occurred
                EXPECT_EQ(memcmp(host, guard, sizeof(guard)), 0);
            }
        }
    }
#endif

    static void free(T* d)
    {
        if(d != nullptr)
        {
#ifdef GOOGLE_TEST
            d = off_guards(d);
#endif
            switch(MODE)
            {
            case memory_mode::host:
            {
                auto status = rocgraph_hipHostFree(d);
                if(status != hipSuccess)
                {
                    throw status;
                }

                break;
            }
            case memory_mode::device:
            {
                // Free device memory
                auto status = rocgraph_hipFree(d);
                if(status != hipSuccess)
                {
                    throw status;
                }
                break;
            }
            case memory_mode::managed:
            {
                // Free device memory
                auto status = rocgraph_hipFreeManaged(d);
                if(status != hipSuccess)
                {
                    throw status;
                }
                break;
            }
            }
        }
    }
};

template <typename T>
using rocgraph_host_allocator = rocgraph_allocator<memory_mode::host, T>;

template <typename T>
using rocgraph_device_allocator = rocgraph_allocator<memory_mode::device, T>;

template <typename T>
using rocgraph_managed_allocator = rocgraph_allocator<memory_mode::managed, T>;

#endif // ROCGRAPH_ALLOCATOR_HPP
