// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_induced_subgraph_result.hpp"

#include "internal/aux/rocgraph_induced_subgraph_result_aux.h"
#include "internal/rocgraph_algorithms.h"

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_induced_subgraph_get_sources(rocgraph_induced_subgraph_result_t* induced_subgraph)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_induced_subgraph_result_t*>(induced_subgraph);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->src_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_induced_subgraph_get_destinations(rocgraph_induced_subgraph_result_t* induced_subgraph)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_induced_subgraph_result_t*>(induced_subgraph);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->dst_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_induced_subgraph_get_edge_weights(rocgraph_induced_subgraph_result_t* induced_subgraph)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_induced_subgraph_result_t*>(induced_subgraph);
    return (internal_pointer->wgt_ == nullptr)
               ? NULL
               : reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
                     internal_pointer->wgt_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_induced_subgraph_get_edge_ids(rocgraph_induced_subgraph_result_t* induced_subgraph)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_induced_subgraph_result_t*>(induced_subgraph);
    return (internal_pointer->edge_ids_ == nullptr)
               ? NULL
               : reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
                     internal_pointer->edge_ids_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t* rocgraph_induced_subgraph_get_edge_type_ids(
    rocgraph_induced_subgraph_result_t* induced_subgraph)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_induced_subgraph_result_t*>(induced_subgraph);
    return (internal_pointer->edge_type_ids_ == nullptr)
               ? NULL
               : reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
                     internal_pointer->edge_type_ids_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t* rocgraph_induced_subgraph_get_subgraph_offsets(
    rocgraph_induced_subgraph_result_t* induced_subgraph)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_induced_subgraph_result_t*>(induced_subgraph);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->subgraph_offsets_->view());
}

extern "C" void
    rocgraph_induced_subgraph_result_free(rocgraph_induced_subgraph_result_t* induced_subgraph)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_induced_subgraph_result_t*>(induced_subgraph);
    delete internal_pointer->src_;
    delete internal_pointer->dst_;
    delete internal_pointer->wgt_;
    delete internal_pointer->edge_ids_;
    delete internal_pointer->edge_type_ids_;
    delete internal_pointer->subgraph_offsets_;
    delete internal_pointer;
}
