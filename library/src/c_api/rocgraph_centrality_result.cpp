// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_centrality_result.hpp"

#include "internal/aux/rocgraph_centrality_result_aux.h"
#include "internal/aux/rocgraph_edge_centrality_result_aux.h"
#include "internal/rocgraph_algorithms.h"

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_centrality_result_get_vertices(rocgraph_centrality_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_centrality_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->vertex_ids_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_centrality_result_get_values(rocgraph_centrality_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_centrality_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->values_->view());
}

size_t rocgraph_centrality_result_get_num_iterations(rocgraph_centrality_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_centrality_result_t*>(result);
    return internal_pointer->num_iterations_;
}

rocgraph_bool rocgraph_centrality_result_converged(rocgraph_centrality_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_centrality_result_t*>(result);
    return internal_pointer->converged_ ? rocgraph_bool_true : rocgraph_bool_false;
}

extern "C" void rocgraph_centrality_result_free(rocgraph_centrality_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_centrality_result_t*>(result);
    delete internal_pointer->vertex_ids_;
    delete internal_pointer->values_;
    delete internal_pointer;
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_edge_centrality_result_get_src_vertices(rocgraph_edge_centrality_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_edge_centrality_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->src_ids_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_edge_centrality_result_get_dst_vertices(rocgraph_edge_centrality_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_edge_centrality_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->dst_ids_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_edge_centrality_result_get_values(rocgraph_edge_centrality_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_edge_centrality_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->values_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_edge_centrality_result_get_edge_ids(rocgraph_edge_centrality_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_edge_centrality_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->edge_ids_->view());
}

extern "C" void rocgraph_edge_centrality_result_free(rocgraph_edge_centrality_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_edge_centrality_result_t*>(result);
    delete internal_pointer->src_ids_;
    delete internal_pointer->dst_ids_;
    delete internal_pointer->values_;
    delete internal_pointer->edge_ids_;
    delete internal_pointer;
}
