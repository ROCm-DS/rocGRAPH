// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_labeling_result.hpp"

#include "internal/aux/rocgraph_labeling_result_aux.h"
#include "internal/rocgraph_labeling_algorithms.h"

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_labeling_result_get_vertices(rocgraph_labeling_result_t* result)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_labeling_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->vertex_ids_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_labeling_result_get_labels(rocgraph_labeling_result_t* result)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_labeling_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->labels_->view());
}

extern "C" void rocgraph_labeling_result_free(rocgraph_labeling_result_t* result)
{
    auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_labeling_result_t*>(result);
    delete internal_pointer->vertex_ids_;
    delete internal_pointer->labels_;
    delete internal_pointer;
}
