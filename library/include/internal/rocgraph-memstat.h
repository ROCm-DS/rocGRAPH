// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*! \file*/
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCGRAPH_MEMSTAT_H
#define ROCGRAPH_MEMSTAT_H

#include "internal/types/rocgraph_status.h"
#include "rocgraph-export.h"

#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

//
// If ROCGRAPH_WITH_MEMSTAT is defined
// then a set of extra routines is offered
// to manage memory with a recording of some traces.
//
#ifdef ROCGRAPH_WITH_MEMSTAT
/*! \ingroup aux_module
   *  \brief Set the memory report filename.
   *
   *  \details
   *  \p rocgraph_memstat_report set the filename to use for the memory report.
   *  This routine is optional, but it must be called before any hip memory operation.
   *  Note that the default memory report filename is 'rocgraph_memstat.json'.
   *  Also note that if any operation occurs before calling this routine, the default filename rocgraph_memstat.json
   *  will be used but renamed after this call.
   *  The content of the memory report summarizes memory operations from the use of the routines
   *  \ref rocgraph_hip_malloc,
   *  \ref rocgraph_hip_free,
   *  \ref rocgraph_hip_host_malloc,
   *  \ref rocgraph_hip_host_free,
   *  \ref rocgraph_hip_host_managed,
   *  \ref rocgraph_hip_free_managed.
   *
   *  @param[in]
   *  filename  the memory report filename.
   *
   *  \retval rocgraph_status_success the operation succeeded.
   *  \retval rocgraph_status_invalid_pointer \p handle filename is an invalid pointer.
   *  \retval rocgraph_status_internal_error an internal error occurred.
   */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_memstat_report(const char* filename);

/*! \ingroup aux_module
   *  \brief Wrap hipFree.
   *
   *  @param[in]
   *  mem  memory pointer
   *  @param[in]
   *  tag  tag to attach to the operation.
   *
   *  \retval error from the related hip operation.
   */
ROCGRAPH_EXPORT
hipError_t rocgraph_hip_free(void* mem, const char* tag);

/*! \ingroup aux_module
   *  \brief Wrap hipMalloc.
   *
   *  @param[in]
   *  mem  pointer of memory pointer
   *  @param[in]
   *  nbytes  number of bytes
   *  @param[in]
   *  tag  tag to attach to the operation
   *
   *  \retval error from the related hip operation
   */
ROCGRAPH_EXPORT
hipError_t rocgraph_hip_malloc(void** mem, size_t nbytes, const char* tag);

/*! \ingroup aux_module
   *  \brief Wrap hipFreeAsync.
   *
   *  @param[in]
   *  mem  memory pointer
   *  @param[in]
   *  stream  the stream to be used by the asynchronous operation
   *  @param[in]
   *  tag  tag to attach to the operation.
   *
   *  \retval error from the related hip operation.
   */
ROCGRAPH_EXPORT
hipError_t rocgraph_hip_free_async(void* mem, hipStream_t stream, const char* tag);

/*! \ingroup aux_module
   *  \brief Wrap hipMallocAsync.
   *
   *  @param[in]
   *  mem  pointer of memory pointer
   *  @param[in]
   *  nbytes  number of bytes
   *  @param[in]
   *  stream  the stream to be used by the asynchronous operation
   *  @param[in]
   *  tag  tag to attach to the operation
   *
   *  \retval error from the related hip operation
   */
ROCGRAPH_EXPORT
hipError_t
    rocgraph_hip_malloc_async(void** mem, size_t nbytes, hipStream_t stream, const char* tag);

/*! \ingroup aux_module
   *  \brief Wrap hipHostFree.
   *
   *  @param[in]
   *  mem  memory pointer
   *  @param[in]
   *  tag  tag to attach to the operation.
   *
   *  \retval error from the related hip operation.
   */
ROCGRAPH_EXPORT
hipError_t rocgraph_hip_host_free(void* mem, const char* tag);

/*! \ingroup aux_module
   *  \brief Wrap hipHostMalloc.
   *
   *  @param[in]
   *  mem  pointer of memory pointer
   *  @param[in]
   *  nbytes  number of bytes
   *  @param[in]
   *  tag  tag to attach to the operation
   *
   *  \retval error from the related hip operation
   */
ROCGRAPH_EXPORT
hipError_t rocgraph_hip_host_malloc(void** mem, size_t nbytes, const char* tag);

/*! \ingroup aux_module
   *  \brief Wrap hipFreeManaged.
   *
   *  @param[in]
   *  mem  memory pointer
   *  @param[in]
   *  tag  tag to attach to the operation.
   *
   *  \retval error from the related hip operation.
   */
ROCGRAPH_EXPORT
hipError_t rocgraph_hip_free_managed(void* mem, const char* tag);

/*! \ingroup aux_module
   *  \brief Wrap hipMallocManaged.
   *
   *  @param[in]
   *  mem  pointer of memory pointer
   *  @param[in]
   *  nbytes  number of bytes
   *  @param[in]
   *  tag  tag to attach to the operation
   *
   *  \retval error from the related hip operation
   */
ROCGRAPH_EXPORT
hipError_t rocgraph_hip_malloc_managed(void** mem, size_t nbytes, const char* tag);

#endif

#ifdef __cplusplus
}
#endif

#endif /* ROCGRAPH_AUXILIARY_H */
