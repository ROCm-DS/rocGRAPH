// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_handle.hpp"
#include "control.h"
#include "internal/aux/rocgraph_handle_aux.h"
#include "memstat.h"
#include "rocgraph-version.h"

ROCGRAPH_KERNEL(1) void init_kernel(){};

const hipDeviceProp_t* _rocgraph_handle::get_properties() const
{
    return &this->m_properties;
}

hipDeviceProp_t* _rocgraph_handle::get_properties()
{
    return &this->m_properties;
}

_rocgraph_handle::_rocgraph_handle(void* raft_handle)
{
    if(raft_handle == nullptr)
    {
        this->m_raft_handle              = new raft::handle_t{};
        this->m_is_raft_handle_allocated = true;
    }
    else
    {
        this->m_raft_handle = reinterpret_cast<raft::handle_t*>(raft_handle);
    }

    // Default device is active device
    THROW_IF_HIP_ERROR(hipGetDevice(&this->m_device));
    THROW_IF_HIP_ERROR(hipGetDeviceProperties(&this->m_properties, this->m_device));

    // Device wavefront size
    this->m_wavefront_size = this->m_properties.warpSize;

#if HIP_VERSION >= 307
    // ASIC revision
    this->m_asic_rev = this->m_properties.asicRevision;
#else
    this->m_asic_rev = 0;
#endif

    // Obtain size for coomv device buffer
    const rocgraph_int nthreads   = this->m_properties.maxThreadsPerBlock;
    const rocgraph_int nprocs     = 2 * this->m_properties.multiProcessorCount;
    const rocgraph_int nblocks    = (nprocs * nthreads - 1) / 256 + 1;
    const size_t       coomv_size = (((sizeof(rocgraph_int) + 16) * nblocks - 1) / 256 + 1) * 256;

    // Allocate device buffer
    this->m_buffer_size = (coomv_size > 1024 * 1024) ? coomv_size : 1024 * 1024;
    THROW_IF_HIP_ERROR(rocgraph_hipMalloc(&this->m_buffer, this->m_buffer_size));

    // Device one
    THROW_IF_HIP_ERROR(rocgraph_hipMalloc(&this->m_sone, sizeof(float)));
    THROW_IF_HIP_ERROR(rocgraph_hipMalloc(&this->m_done, sizeof(double)));

    // Execute empty kernel for initialization
    THROW_IF_HIPLAUNCHKERNELGGL_ERROR(init_kernel, dim3(1), dim3(1), 0, this->get_stream());

    // Execute memset for initialization
    THROW_IF_HIP_ERROR(hipMemsetAsync(this->m_sone, 0, sizeof(float), this->get_stream()));
    THROW_IF_HIP_ERROR(hipMemsetAsync(this->m_done, 0, sizeof(double), this->get_stream()));

    static constexpr float  hsone = 1.0f;
    static constexpr double hdone = 1.0;

    THROW_IF_HIP_ERROR(hipMemcpyAsync(
        this->m_sone, &hsone, sizeof(float), hipMemcpyHostToDevice, this->get_stream()));
    THROW_IF_HIP_ERROR(hipMemcpyAsync(
        this->m_done, &hdone, sizeof(double), hipMemcpyHostToDevice, this->get_stream()));

    // Wait for device transfer to finish
    THROW_IF_HIP_ERROR(hipStreamSynchronize(this->get_stream()));
}

_rocgraph_handle::~_rocgraph_handle()
{
    if(this->m_is_raft_handle_allocated)
    {
        delete this->m_raft_handle;
        this->m_raft_handle = nullptr;
    }

    PRINT_IF_HIP_ERROR(rocgraph_hipFree(this->m_buffer));
    PRINT_IF_HIP_ERROR(rocgraph_hipFree(this->m_sone));
    PRINT_IF_HIP_ERROR(rocgraph_hipFree(this->m_done));
}

const raft::handle_t* _rocgraph_handle::get_raft_handle() const
{
    return this->m_raft_handle;
}
raft::handle_t* _rocgraph_handle::get_raft_handle()
{
    return this->m_raft_handle;
}
int32_t _rocgraph_handle::get_wavefront_size() const
{
    return this->m_wavefront_size;
}
void* _rocgraph_handle::get_buffer()
{
    return this->m_buffer;
}
const void* _rocgraph_handle::get_buffer() const
{
    return this->m_buffer;
}
size_t _rocgraph_handle::get_buffer_size() const
{
    return this->m_buffer_size;
}

rocgraph_status _rocgraph_handle::set_stream(hipStream_t user_stream)
{
    // TODO check if stream is valid
    this->m_stream = user_stream;

    return rocgraph_status_success;
}

hipStream_t _rocgraph_handle::get_stream()
{
    return this->m_stream;
}

const hipStream_t _rocgraph_handle::get_stream() const
{
    return this->m_stream;
}

rocgraph_status _rocgraph_handle::set_pointer_mode(rocgraph_pointer_mode user_mode)
{
    // TODO check if stream is valid
    this->m_pointer_mode = user_mode;

    return rocgraph_status_success;
}

rocgraph_pointer_mode _rocgraph_handle::get_pointer_mode() const
{
    return this->m_pointer_mode;
}

extern "C" rocgraph_status rocgraph_create_handle(rocgraph_handle* handle, void* raft_handle)
try
{
    ROCGRAPH_CHECKARG_POINTER(0, handle);
    *handle = new rocgraph_handle_t(raft_handle);
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

extern "C" rocgraph_status rocgraph_destroy_handle(rocgraph_handle handle)
try
{
    ROCGRAPH_CHECKARG_HANDLE(0, handle);
    delete handle;
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

extern "C" rocgraph_status rocgraph_set_pointer_mode(rocgraph_handle_t*    handle,
                                                     rocgraph_pointer_mode mode)
try
{
    ROCGRAPH_CHECKARG_HANDLE(0, handle);
    ROCGRAPH_CHECKARG_ENUM(1, mode);
    RETURN_IF_ROCGRAPH_ERROR(handle->set_pointer_mode(mode));
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

extern "C" rocgraph_status rocgraph_get_pointer_mode(const rocgraph_handle_t* handle,
                                                     rocgraph_pointer_mode*   mode)
try
{
    ROCGRAPH_CHECKARG_HANDLE(0, handle);
    ROCGRAPH_CHECKARG_POINTER(1, mode);
    mode[0] = handle->get_pointer_mode();
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

extern "C" rocgraph_status rocgraph_set_stream(rocgraph_handle_t* handle, hipStream_t stream_id)
try
{
    ROCGRAPH_CHECKARG_HANDLE(0, handle);
    RETURN_IF_ROCGRAPH_ERROR(handle->set_stream(stream_id));
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

extern "C" rocgraph_status rocgraph_get_stream(const rocgraph_handle_t* handle,
                                               hipStream_t*             stream_id)
try
{
    ROCGRAPH_CHECKARG_HANDLE(0, handle);
    stream_id[0] = handle->get_stream();
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

extern "C" rocgraph_status rocgraph_get_version(const rocgraph_handle_t* handle, int* version)
try
{
    ROCGRAPH_CHECKARG_HANDLE(0, handle);
    *version
        = ROCGRAPH_VERSION_MAJOR * 100000 + ROCGRAPH_VERSION_MINOR * 100 + ROCGRAPH_VERSION_PATCH;

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

extern "C" rocgraph_status rocgraph_get_git_rev(const rocgraph_handle_t* handle, char* rev)
try
{
    ROCGRAPH_CHECKARG_HANDLE(0, handle);
    ROCGRAPH_CHECKARG_POINTER(1, rev);

    static constexpr char v[] = TO_STR(ROCGRAPH_VERSION_TWEAK);

    memcpy(rev, v, sizeof(v));

    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

rocgraph_status rocgraph_handle_get_rank(const rocgraph_handle_t* handle, int32_t* p_rank)
try
{
    ROCGRAPH_CHECKARG_HANDLE(0, handle);
    ROCGRAPH_CHECKARG_POINTER(1, p_rank);
    auto  internal = handle;
    auto& comm     = internal->get_raft_handle()->get_comms();
    p_rank[0]      = static_cast<int32_t>(comm.get_rank());
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}

extern "C" rocgraph_status rocgraph_handle_get_comm_size(const rocgraph_handle_t* handle,
                                                         int32_t*                 p_comm_size)
try
{
    ROCGRAPH_CHECKARG_HANDLE(0, handle);
    ROCGRAPH_CHECKARG_POINTER(1, p_comm_size);
    auto  internal = handle;
    auto& comm     = internal->get_raft_handle()->get_comms();
    p_comm_size[0] = static_cast<int32_t>(comm.get_size());
    return rocgraph_status_success;
}
catch(...)
{
    RETURN_ROCGRAPH_EXCEPTION();
}
