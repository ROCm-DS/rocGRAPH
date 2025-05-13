// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_core_result.hpp"

#include "internal/aux/rocgraph_core_result_aux.h"
#include "internal/aux/rocgraph_k_core_result_aux.h"
#include "internal/aux/rocgraph_type_erased_device_array_aux.h"
#include "internal/rocgraph_algorithms.h"

extern "C" rocgraph_status
    rocgraph_core_result_create(const rocgraph_handle_t*                  handle,
                                rocgraph_type_erased_device_array_view_t* vertices,
                                rocgraph_type_erased_device_array_view_t* core_numbers,
                                rocgraph_core_result_t**                  core_result,
                                rocgraph_error_t**                        error)
{
    rocgraph_status status{rocgraph_status_success};

    rocgraph::c_api::rocgraph_type_erased_device_array_t* vertices_copy;
    rocgraph::c_api::rocgraph_type_erased_device_array_t* core_numbers_copy;

    status = rocgraph_type_erased_device_array_create_from_view(
        handle,
        vertices,
        reinterpret_cast<rocgraph_type_erased_device_array_t**>(&vertices_copy),
        error);
    if(status == rocgraph_status_success)
    {
        status = rocgraph_type_erased_device_array_create_from_view(
            handle,
            core_numbers,
            reinterpret_cast<rocgraph_type_erased_device_array_t**>(&core_numbers_copy),
            error);

        if(status == rocgraph_status_success)
        {
            auto internal_pointer
                = new rocgraph::c_api::rocgraph_core_result_t{vertices_copy, core_numbers_copy};
            *core_result = reinterpret_cast<rocgraph_core_result_t*>(internal_pointer);
        }
    }
    return status;
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_core_result_get_vertices(rocgraph_core_result_t* result)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_core_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->vertex_ids_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_core_result_get_core_numbers(rocgraph_core_result_t* result)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_core_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->core_numbers_->view());
}

extern "C" void rocgraph_core_result_free(rocgraph_core_result_t* result)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_core_result_t*>(result);
    delete internal_pointer->vertex_ids_;
    delete internal_pointer->core_numbers_;
    delete internal_pointer;
}

rocgraph_type_erased_device_array_view_t*
    rocgraph_k_core_result_get_src_vertices(rocgraph_k_core_result_t* result)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_k_core_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->src_vertices_->view());
}

rocgraph_type_erased_device_array_view_t*
    rocgraph_k_core_result_get_dst_vertices(rocgraph_k_core_result_t* result)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_k_core_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->dst_vertices_->view());
}

rocgraph_type_erased_device_array_view_t*
    rocgraph_k_core_result_get_weights(rocgraph_k_core_result_t* result)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_k_core_result_t*>(result);
    return (internal_pointer->weights_ == nullptr)
               ? NULL
               : reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
                     internal_pointer->weights_->view());
}

void rocgraph_k_core_result_free(rocgraph_k_core_result_t* result)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_k_core_result_t*>(result);
    delete internal_pointer->src_vertices_;
    delete internal_pointer->dst_vertices_;
    delete internal_pointer->weights_;
    delete internal_pointer;
}
