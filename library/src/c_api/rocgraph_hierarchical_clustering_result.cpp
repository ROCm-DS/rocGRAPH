// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_hierarchical_clustering_result.hpp"

#include "internal/aux/rocgraph_hierarchical_clustering_result_aux.h"
#include "internal/rocgraph_community_algorithms.h"

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_hierarchical_clustering_result_get_vertices(
        rocgraph_hierarchical_clustering_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_hierarchical_clustering_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->vertices_->view());
}

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_hierarchical_clustering_result_get_clusters(
        rocgraph_hierarchical_clustering_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_hierarchical_clustering_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->clusters_->view());
}

extern "C" double rocgraph_hierarchical_clustering_result_get_modularity(
    rocgraph_hierarchical_clustering_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_hierarchical_clustering_result_t*>(result);
    return internal_pointer->modularity;
}

extern "C" void
    rocgraph_hierarchical_clustering_result_free(rocgraph_hierarchical_clustering_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_hierarchical_clustering_result_t*>(result);
    delete internal_pointer->vertices_;
    delete internal_pointer->clusters_;
    delete internal_pointer;
}
