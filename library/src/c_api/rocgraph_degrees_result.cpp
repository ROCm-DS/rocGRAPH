// Copyright (C) 2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_degrees_result.hpp"

#include "internal/aux/rocgraph_degrees_result_aux.h"
#include "internal/rocgraph_graph_functions.h"

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_degrees_result_get_vertices(rocgraph_degrees_result_t* degrees_result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_degrees_result_t*>(degrees_result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->vertex_ids_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_degrees_result_get_in_degrees(rocgraph_degrees_result_t* degrees_result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_degrees_result_t*>(degrees_result);
    return internal_pointer->in_degrees_ == nullptr
               ? nullptr
               : reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
                     internal_pointer->in_degrees_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_degrees_result_get_out_degrees(rocgraph_degrees_result_t* degrees_result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_degrees_result_t*>(degrees_result);
    return internal_pointer->out_degrees_ != nullptr
               ? reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
                     internal_pointer->out_degrees_->view())
           : internal_pointer->is_symmetric
               ? reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
                     internal_pointer->in_degrees_->view())
               : nullptr;
}

extern "C" void rocgraph_degrees_result_free(rocgraph_degrees_result_t* degrees_result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_degrees_result_t*>(degrees_result);
    delete internal_pointer->vertex_ids_;
    delete internal_pointer->in_degrees_;
    delete internal_pointer->out_degrees_;
    delete internal_pointer;
}
