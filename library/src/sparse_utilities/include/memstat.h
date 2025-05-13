/*! \file */

/*
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <hip/hip_runtime_api.h>

//
// This section is conditional to the definition
// of ROCGRAPH_WITH_MEMSTAT
//
#ifndef ROCGRAPH_WITH_MEMSTAT

#define rocgraph_hipMalloc(p_, nbytes_) hipMalloc(p_, nbytes_)
#define rocgraph_hipFree(p_) hipFree(p_)

// if hip version is atleast 5.3.0 hipMallocAsync and hipFreeAsync are defined
#if HIP_VERSION >= 50300000
#define rocgraph_hipMallocAsync(p_, nbytes_, stream_) hipMallocAsync(p_, nbytes_, stream_)
#define rocgraph_hipFreeAsync(p_, stream_) hipFreeAsync(p_, stream_)
#else
#define rocgraph_hipMallocAsync(p_, nbytes_, stream_) hipMalloc(p_, nbytes_)
#define rocgraph_hipFreeAsync(p_, stream_) hipFree(p_)
#endif

#define rocgraph_hipHostMalloc(p_, nbytes_) hipHostMalloc(p_, nbytes_)
#define rocgraph_hipHostFree(p_) hipHostFree(p_)

#define rocgraph_hipMallocManaged(p_, nbytes_) hipMallocManaged(p_, nbytes_)
#define rocgraph_hipFreeManaged(p_) hipFree(p_)

#else

#include "rocgraph-auxiliary.h"

#define ROCGRAPH_HIP_SOURCE_MSG(msg_) #msg_
#define ROCGRAPH_HIP_SOURCE_TAG(msg_) __FILE__ " " ROCGRAPH_HIP_SOURCE_MSG(msg_)

#define rocgraph_hipMalloc(p_, nbytes_) \
    rocgraph_hip_malloc((void**)(p_), (nbytes_), ROCGRAPH_HIP_SOURCE_TAG(__LINE__))

#define rocgraph_hipFree(p_) rocgraph_hip_free((void**)(p_), ROCGRAPH_HIP_SOURCE_TAG(__LINE__))

#define rocgraph_hipMallocAsync(p_, nbytes_, stream_) \
    rocgraph_hip_malloc_async((void**)(p_), (nbytes_), stream_, ROCGRAPH_HIP_SOURCE_TAG(__LINE__))

#define rocgraph_hipFreeAsync(p_, stream_) \
    rocgraph_hip_free_async((void**)(p_), stream_, ROCGRAPH_HIP_SOURCE_TAG(__LINE__))

#define rocgraph_hipHostMalloc(p_, nbytes_) \
    rocgraph_hip_host_malloc((void**)(p_), (nbytes_), ROCGRAPH_HIP_SOURCE_TAG(__LINE__))

#define rocgraph_hipHostFree(p_) \
    rocgraph_hip_host_free((void**)(p_), ROCGRAPH_HIP_SOURCE_TAG(__LINE__))

#define rocgraph_hipMallocManaged(p_, nbytes_) \
    rocgraph_hip_malloc_managed((void**)(p_), (nbytes_), ROCGRAPH_HIP_SOURCE_TAG(__LINE__))

#define rocgraph_hipFreeManaged(p_) \
    rocgraph_hip_free_managed((void**)(p_), ROCGRAPH_HIP_SOURCE_TAG(__LINE__))

#endif
