// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*!\file*/
/*
 * Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 */
/*
 * Copyright (C) 2022-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "internal/types/rocgraph_handle_t.h"
#include "rocgraph-export.h"
#include <hip/hip_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif

/** @ingroup aux_module
   *  @brief Create a rocgraph handle
   *
   *  @details
   *  @p rocgraph_create_handle creates the rocGRAPH library context. It must be
   *  initialized before any other rocGRAPH API function is invoked and must be passed to
   *  all subsequent library function calls. The handle should be destroyed at the end
   *  using rocgraph_destroy_handle().
   *
   *  @param[out]
   *  handle  the pointer to the handle to the rocGRAPH library context.
   *
   *  @param[in]  raft_handle   Handle for accessing resources
   *                            If NULL, we will create a raft handle
   *                            internally
   *
   *  @retval rocgraph_status_success the initialization succeeded.
   *  @retval rocgraph_status_invalid_handle @p handle pointer is invalid.
   *  @retval rocgraph_status_internal_error an internal error occurred.
   */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_create_handle(rocgraph_handle_t** handle, void* raft_handle);

/** @ingroup aux_module
   *  @brief Destroy a rocgraph handle
   *
   *  @details
   *  @p rocgraph_destroy_handle destroys the rocGRAPH library context and releases all
   *  resources used by the rocGRAPH library.
   *
   *  @param[in]
   *  handle  the handle to the rocGRAPH library context.
   *
   *  @retval rocgraph_status_success the operation completed successfully.
   *  @retval rocgraph_status_invalid_handle @p handle is invalid.
   *  @retval rocgraph_status_internal_error an internal error occurred.
   */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_destroy_handle(rocgraph_handle_t* handle);

/** @ingroup aux_module
   *  @brief Specify user defined HIP stream
   *
   *  @details
   *  @p rocgraph_set_stream specifies the stream to be used by the rocGRAPH library
   *  context and all subsequent function calls.
   *
   *  @param[inout]
   *  handle  the handle to the rocGRAPH library context.
   *  @param[in]
   *  stream  the stream to be used by the rocGRAPH library context.
   *
   *  @retval rocgraph_status_success the operation completed successfully.
   *  @retval rocgraph_status_invalid_handle @p handle is invalid.
   *
   *  @par Example
   *  This example illustrates, how a user defined stream can be used in rocGRAPH.
   *  @code{.c}
   *      // Create rocGRAPH handle
   *      rocgraph_handle handle;
   *      rocgraph_create_handle(&handle);
   *
   *      // Create stream
   *      hipStream_t stream;
   *      hipStreamCreate(&stream);
   *
   *      // Set stream to rocGRAPH handle
   *      rocgraph_set_stream(handle, stream);
   *
   *      // Do some work
   *      // ...
   *
   *      // Clean up
   *      rocgraph_destroy_handle(handle);
   *      hipStreamDestroy(stream);
   *  @endcode
   */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_set_stream(rocgraph_handle_t* handle, hipStream_t stream);

/** @ingroup aux_module
   *  @brief Get current stream from library context
   *
   *  @details
   *  @p rocgraph_get_stream gets the rocGRAPH library context stream which is currently
   *  used for all subsequent function calls.
   *
   *  @param[in]
   *  handle the handle to the rocGRAPH library context.
   *  @param[out]
   *  stream the stream currently used by the rocGRAPH library context.
   *
   *  @retval rocgraph_status_success the operation completed successfully.
   *  @retval rocgraph_status_invalid_handle @p handle is invalid.
   */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_get_stream(const rocgraph_handle_t* handle, hipStream_t* stream);

/** @ingroup aux_module
   *  @brief Specify pointer mode
   *
   *  @details
   *  @p rocgraph_set_pointer_mode specifies the pointer mode to be used by the rocGRAPH
   *  library context and all subsequent function calls. By default, all values are passed
   *  by reference on the host. Valid pointer modes are @ref rocgraph_pointer_mode_host
   *  or @p rocgraph_pointer_mode_device.
   *
   *  @param[in]
   *  handle          the handle to the rocGRAPH library context.
   *  @param[in]
   *  pointer_mode    the pointer mode to be used by the rocGRAPH library context.
   *
   *  @retval rocgraph_status_success the operation completed successfully.
   *  @retval rocgraph_status_invalid_handle @p handle is invalid.
   */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_set_pointer_mode(rocgraph_handle_t*    handle,
                                          rocgraph_pointer_mode pointer_mode);

/** @ingroup aux_module
   *  @brief Get current pointer mode from library context
   *
   *  @details
   *  @p rocgraph_get_pointer_mode gets the rocGRAPH library context pointer mode which
   *  is currently used for all subsequent function calls.
   *
   *  @param[in]
   *  handle          the handle to the rocGRAPH library context.
   *  @param[out]
   *  pointer_mode    the pointer mode that is currently used by the rocGRAPH library
   *                  context.
   *
   *  @retval rocgraph_status_success the operation completed successfully.
   *  @retval rocgraph_status_invalid_handle @p handle is invalid.
   */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_get_pointer_mode(const rocgraph_handle_t* handle,
                                          rocgraph_pointer_mode*   pointer_mode);

/** @ingroup aux_module
   *  @brief Get rocGRAPH version
   *
   *  @details
   *  @p rocgraph_get_version gets the rocGRAPH library version number.
   *  - patch = version % 100
   *  - minor = version / 100 % 1000
   *  - major = version / 100000
   *
   *  @param[in]
   *  handle  the handle to the rocGRAPH library context.
   *  @param[out]
   *  version the version number of the rocGRAPH library.
   *
   *  @retval rocgraph_status_success the operation completed successfully.
   *  @retval rocgraph_status_invalid_handle @p handle is invalid.
   */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_get_version(const rocgraph_handle_t* handle, int* version);

/** @ingroup aux_module
   *  @brief Get rocGRAPH git revision
   *
   *  @details
   *  @p rocgraph_get_git_rev gets the rocGRAPH library git commit revision (SHA-1).
   *
   *  @param[in]
   *  handle  the handle to the rocGRAPH library context.
   *  @param[out]
   *  rev     the git commit revision (SHA-1).
   *
   *  @retval rocgraph_status_success the operation completed successfully.
   *  @retval rocgraph_status_invalid_handle @p handle is invalid.
   */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_get_git_rev(const rocgraph_handle_t* handle, char* rev);

/**
   * @brief get rank from resource handle
   *
   * If the resource handle has been configured for multi-gpu, this will return
   * the rank for this worker.  If the resource handle has not been configured for
   * multi-gpu this will always return 0.
   *
   * @param [in]  handle          Handle for accessing resources
   * @param [out] p_comm_size     Pointer of the output rank value.
   * @retval rocgraph_status_invalid_handle @p handle is invalid.
   * @retval rocgraph_status_invalid_pointer @p p_rank is invalid.
   */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_handle_get_rank(const rocgraph_handle_t* handle, int32_t* p_rank);

/**
   * @brief get comm_size from resource handle
   *
   * If the resource handle has been configured for multi-gpu, this will return
   * the comm_size for this cluster.  If the resource handle has not been configured for
   * multi-gpu this will always return 1.
   *
   * @param [in]  handle          Handle for accessing resources
   * @param [out] p_comm_size     Pointer of the output comm size value.
   * @retval rocgraph_status_success the operation completed successfully.
   * @retval rocgraph_status_invalid_handle @p handle is invalid.
   * @retval rocgraph_status_invalid_pointer @p p_comm_size is invalid.
   */
ROCGRAPH_EXPORT
rocgraph_status rocgraph_handle_get_comm_size(const rocgraph_handle_t* handle,
                                              int32_t*                 p_comm_size);

#ifdef __cplusplus
}
#endif
