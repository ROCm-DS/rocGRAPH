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

#include "testing_rocgraph_edge_betweenness_centrality.hpp"
#include "rocgraph/rocgraph.h"
#include "testing.hpp"

#include "rocgraph_clients_create_test_graph.hpp"
#include "rocgraph_test.hpp"

namespace
{
    template <typename weight_t, typename vertex_t>
    void generic_edge_betweenness_centrality_test(vertex_t*     h_src,
                                                  vertex_t*     h_dst,
                                                  weight_t*     h_wgt,
                                                  vertex_t*     h_seeds,
                                                  weight_t*     h_result,
                                                  size_t        num_vertices,
                                                  size_t        num_edges,
                                                  size_t        num_seeds,
                                                  rocgraph_bool store_transposed,
                                                  size_t        num_vertices_to_sample)
    {
        rocgraph_error_t* ret_error;

        rocgraph_handle_t*                        p_handle   = nullptr;
        rocgraph_graph_t*                         graph      = nullptr;
        rocgraph_edge_centrality_result_t*        result     = nullptr;
        rocgraph_rng_state_t*                     rng_state  = nullptr;
        rocgraph_type_erased_device_array_t*      seeds      = nullptr;
        rocgraph_type_erased_device_array_view_t* seeds_view = nullptr;

        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_rng_state_create(p_handle, 0, &rng_state, &ret_error),
                                     ret_error);

        rocgraph_clients_create_test_graph(p_handle,
                                           h_src,
                                           h_dst,
                                           h_wgt,
                                           num_edges,
                                           store_transposed,
                                           rocgraph_bool_false,
                                           rocgraph_bool_false,
                                           &graph,
                                           &ret_error);

        if(h_seeds == nullptr)
        {
            CHECK_ROCGRAPH_SUCCESS_ERROR(
                rocgraph_select_random_vertices(
                    p_handle, graph, rng_state, num_vertices_to_sample, &seeds, &ret_error),
                ret_error);

            seeds_view = rocgraph_type_erased_device_array_view(seeds);
        }
        else
        {
            CHECK_ROCGRAPH_SUCCESS_ERROR(
                rocgraph_type_erased_device_array_create(
                    p_handle, num_seeds, rocgraph_data_type_id_int32, &seeds, &ret_error),
                ret_error);

            seeds_view = rocgraph_type_erased_device_array_view(seeds);
            CHECK_ROCGRAPH_SUCCESS_ERROR(
                rocgraph_type_erased_device_array_view_copy_from_host(
                    p_handle, seeds_view, (rocgraph_byte_t*)h_seeds, &ret_error),
                ret_error);
        }

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_edge_betweenness_centrality(p_handle,
                                                                          graph,
                                                                          seeds_view,
                                                                          rocgraph_bool_false,
                                                                          rocgraph_bool_false,
                                                                          &result,
                                                                          &ret_error),
                                     ret_error);

        rocgraph_type_erased_device_array_view_t* srcs;
        rocgraph_type_erased_device_array_view_t* dsts;
        rocgraph_type_erased_device_array_view_t* centralities;

        srcs         = rocgraph_edge_centrality_result_get_src_vertices(result);
        dsts         = rocgraph_edge_centrality_result_get_dst_vertices(result);
        centralities = rocgraph_edge_centrality_result_get_values(result);

        size_t num_local_edges = rocgraph_type_erased_device_array_view_size(srcs);

        std::vector<vertex_t> h_rocgraph_src(num_local_edges);
        std::vector<vertex_t> h_rocgraph_dst(num_local_edges);
        std::vector<weight_t> h_centralities(num_local_edges);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_rocgraph_src.data(), srcs, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_rocgraph_dst.data(), dsts, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_centralities.data(), centralities, &ret_error),
            ret_error);

        std::vector<weight_t> M(num_vertices * num_vertices, 0.0);

        for(size_t i = 0; i < num_edges; ++i)
        {
            M[h_src[i] + num_vertices * h_dst[i]] = h_result[i];
        }

        for(size_t i = 0; i < num_local_edges; ++i)
        {
            ROCGRAPH_CLIENTS_EXPECT_NEAR_TOLERANCE(
                M[h_rocgraph_src[i] + num_vertices * h_rocgraph_dst[i]], h_centralities[i], 0.001);
        }

        rocgraph_edge_centrality_result_free(result);
        rocgraph_sg_graph_free(graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    void EdgeBetweennessCentrality(const Arguments& arg)
    {

        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 16;
        size_t num_vertices = 6;

        vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
        weight_t h_wgt[]    = {0.1f,
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
        weight_t h_result[] = {0,
                               2,
                               3,
                               1.83333,
                               2,
                               2,
                               3,
                               2,
                               3.16667,
                               2.83333,
                               4.33333,
                               0,
                               2,
                               2.83333,
                               3.66667,
                               2.33333};

        // double epsilon        = 1e-6;
        // size_t max_iterations = 200;

        // Eigenvector centrality wants store_transposed = rocgraph_bool_true
        generic_edge_betweenness_centrality_test<weight_t, vertex_t>(h_src,
                                                                     h_dst,
                                                                     h_wgt,
                                                                     nullptr,
                                                                     h_result,
                                                                     num_vertices,
                                                                     num_edges,
                                                                     0,
                                                                     rocgraph_bool_true,
                                                                     5);
    }

} // namespace

template <typename T>
void testing_rocgraph_edge_betweenness_centrality_bad_arg(const Arguments& arg)
{
#ifdef TO_FILL
    const rocgraph_handle_t*                        handle{};
    rocgraph_graph_t*                               graph{};
    const rocgraph_type_erased_device_array_view_t* vertex_list{};
    rocgraph_bool                                   normalized{};
    rocgraph_bool                                   do_expensive_check{};
    rocgraph_edge_centrality_result_t**             result{};
    rocgraph_error_t**                              error{};
    auto                                            ret = rocgraph_edge_betweenness_centrality(
        handle, graph, vertex_list, normalized, do_expensive_check, result, error);
    if(ret != rocgraph_status_success)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_internal_error);
    }
    CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
#endif
}

template <typename T>
void testing_rocgraph_edge_betweenness_centrality(const Arguments& arg)
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

#define INSTANTIATE(TYPE)                                                     \
    template void testing_rocgraph_edge_betweenness_centrality_bad_arg<TYPE>( \
        const Arguments& arg);                                                \
    template void testing_rocgraph_edge_betweenness_centrality<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_edge_betweenness_centrality_extra(const Arguments& arg)
{
    testing_dispatch_extra(arg, "EdgeBetweennessCentrality", EdgeBetweennessCentrality);
}
