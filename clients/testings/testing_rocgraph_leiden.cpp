// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

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

#include "testing_rocgraph_leiden.hpp"
#include "rocgraph/rocgraph.h"
#include "rocgraph_clients_create_test_graph.hpp"
#include "rocgraph_test.hpp"
#include "testing.hpp"

namespace
{

    template <typename weight_t, typename edge_t, typename vertex_t>
    void generic_leiden_test(vertex_t*     h_src,
                             vertex_t*     h_dst,
                             weight_t*     h_wgt,
                             vertex_t*     h_result,
                             weight_t      expected_modularity,
                             size_t        num_vertices,
                             size_t        num_edges,
                             size_t        max_level,
                             double        resolution,
                             double        theta,
                             rocgraph_bool store_transposed)
    {
        rocgraph_error_t* ret_error;

        rocgraph_handle_t*                         p_handle    = nullptr;
        rocgraph_rng_state_t*                      p_rng_state = nullptr;
        rocgraph_graph_t*                          p_graph     = nullptr;
        rocgraph_hierarchical_clustering_result_t* p_result    = nullptr;

        rocgraph_data_type_id vertex_tid    = rocgraph_data_type_id_int32;
        rocgraph_data_type_id edge_tid      = rocgraph_data_type_id_int32;
        rocgraph_data_type_id weight_tid    = rocgraph_data_type_id_float32;
        rocgraph_data_type_id edge_id_tid   = rocgraph_data_type_id_int32;
        rocgraph_data_type_id edge_type_tid = rocgraph_data_type_id_int32;

        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_rng_state_create(p_handle, 0, &p_rng_state, &ret_error), ret_error);

        rocgraph_clients_create_sg_test_graph(p_handle,
                                              vertex_tid,
                                              edge_tid,
                                              h_src,
                                              h_dst,
                                              weight_tid,
                                              h_wgt,
                                              edge_type_tid,
                                              nullptr,
                                              edge_id_tid,
                                              nullptr,
                                              num_edges,
                                              store_transposed,
                                              rocgraph_bool_false,
                                              rocgraph_bool_false,
                                              rocgraph_bool_false,
                                              &p_graph,
                                              &ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_leiden(p_handle,
                                                     p_rng_state,
                                                     p_graph,
                                                     max_level,
                                                     resolution,
                                                     theta,
                                                     rocgraph_bool_false,
                                                     &p_result,
                                                     &ret_error),
                                     ret_error);

        rocgraph_type_erased_device_array_view_t* vertices;
        rocgraph_type_erased_device_array_view_t* clusters;

        vertices          = rocgraph_hierarchical_clustering_result_get_vertices(p_result);
        clusters          = rocgraph_hierarchical_clustering_result_get_clusters(p_result);
        double modularity = rocgraph_hierarchical_clustering_result_get_modularity(p_result);

        std::vector<vertex_t> h_vertices(num_vertices);
        std::vector<edge_t>   h_clusters(num_vertices);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_vertices.data(), vertices, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_clusters.data(), clusters, &ret_error),
            ret_error);

        const double double_expected_modularity = expected_modularity;
        ROCGRAPH_CLIENTS_EXPECT_NEAR_TOLERANCE(modularity, double_expected_modularity, 0.001);

        rocgraph_hierarchical_clustering_result_free(p_result);

        rocgraph_sg_graph_free(p_graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    void Leiden(const Arguments& arg)
    {

        using vertex_t = int32_t;
        using edge_t   = int32_t;
        using weight_t = float;

        size_t   num_edges    = 16;
        size_t   num_vertices = 6;
        size_t   max_level    = 10;
        weight_t resolution   = 1.0;
        weight_t theta        = 1.0;

        vertex_t h_src[]             = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_dst[]             = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
        weight_t h_wgt[]             = {0.1f,
                                        2.1f,
                                        1.1f,
                                        5.1f,
                                        3.1f,
                                        4.1f,
                                        7.2f,
                                        3.2f,
                                        0.1f,
                                        2.1f,
                                        1.1f,
                                        5.1f,
                                        3.1f,
                                        4.1f,
                                        7.2f,
                                        3.2f};
        vertex_t h_result[]          = {0, 0, 0, 1, 1, 1};
        weight_t expected_modularity = 0.215969;

        // Louvain wants store_transposed = rocgraph_bool_false
        generic_leiden_test<weight_t, edge_t, vertex_t>(h_src,
                                                        h_dst,
                                                        h_wgt,
                                                        h_result,
                                                        expected_modularity,
                                                        num_vertices,
                                                        num_edges,
                                                        max_level,
                                                        resolution,
                                                        theta,
                                                        rocgraph_bool_false);
    }

    void LeidenNoWeights(const Arguments& arg)
    {

        using vertex_t        = int32_t;
        using edge_t          = int32_t;
        using weight_t        = float;
        size_t   num_edges    = 16;
        size_t   num_vertices = 6;
        size_t   max_level    = 10;
        weight_t resolution   = 1.0;
        weight_t theta        = 1.0;

        vertex_t h_src[]             = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_dst[]             = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_result[]          = {1, 1, 1, 2, 0, 0};
        weight_t expected_modularity = 0.125;

        // Louvain wants store_transposed = rocgraph_bool_false
        generic_leiden_test<weight_t, edge_t, vertex_t>(h_src,
                                                        h_dst,
                                                        nullptr,
                                                        h_result,
                                                        expected_modularity,
                                                        num_vertices,
                                                        num_edges,
                                                        max_level,
                                                        resolution,
                                                        theta,
                                                        rocgraph_bool_false);
    }

} // namespace

template <typename T>
void testing_rocgraph_leiden_bad_arg(const Arguments& arg)
{
#ifdef TO_FILL

    const rocgraph_handle_t*                    handle{};
    rocgraph_rng_state_t*                       rng_state{};
    rocgraph_graph_t*                           graph{};
    size_t                                      max_level{};
    double                                      resolution{};
    double                                      theta{};
    rocgraph_bool                               do_expensive_check{};
    rocgraph_hierarchical_clustering_result_t** result{};
    rocgraph_error_t**                          error{};
    auto                                        ret = rocgraph_leiden(
        handle, rng_state, graph, max_level, resolution, theta, do_expensive_check, result, error);
    if(ret != rocgraph_status_success)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_internal_error);
    }
    CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
#endif
}

template <typename T>
void testing_rocgraph_leiden(const Arguments& arg)
{
#ifdef TO_FILL

    //
    // Unit check.
    //
    if(arg.unit_check)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
    }

    if(arg.timing)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        //
        // Warm-up
        //
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
        }

        double gpu_time_used = get_time_us();
        {
            //
            // Performance run
            //
            for(int iter = 0; iter < number_hot_calls; ++iter)
            {
            }
        }
        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        display_timing_info(display_key_t::time_ms, get_gpu_time_msec(gpu_time_used));
    }
#endif
}

#define INSTANTIATE(TYPE)                                                      \
    template void testing_rocgraph_leiden_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_rocgraph_leiden<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_leiden_extra(const Arguments& arg)
{
    testing_dispatch_extra(arg, "Leiden", Leiden, "LeidenNoWeights", LeidenNoWeights);
}
