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

#include "testing_rocgraph_betweenness_centrality.hpp"
#include "rocgraph/rocgraph.h"
#include "rocgraph_clients_create_test_graph.hpp"
#include "rocgraph_test.hpp"
#include "testing.hpp"

namespace
{
    template <typename weight_t, typename vertex_t>
    void generic_betweenness_centrality_test(vertex_t*     h_src,
                                             vertex_t*     h_dst,
                                             weight_t*     h_wgt,
                                             vertex_t*     h_seeds,
                                             weight_t*     h_result,
                                             size_t        num_vertices,
                                             size_t        num_edges,
                                             size_t        num_seeds,
                                             rocgraph_bool store_transposed,
                                             rocgraph_bool is_symmetric,
                                             rocgraph_bool normalized,
                                             rocgraph_bool include_endpoints,
                                             size_t        num_vertices_to_sample)
    {

        rocgraph_error_t* ret_error;

        rocgraph_handle_t*                        p_handle   = nullptr;
        rocgraph_graph_t*                         p_graph    = nullptr;
        rocgraph_centrality_result_t*             p_result   = nullptr;
        rocgraph_rng_state_t*                     rng_state  = nullptr;
        rocgraph_type_erased_device_array_t*      seeds      = nullptr;
        rocgraph_type_erased_device_array_view_t* seeds_view = nullptr;

        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_rng_state_create(p_handle, 0, &rng_state, &ret_error),
                                     ret_error);
        rocgraph_clients_create_test_graph<weight_t, vertex_t>(p_handle,
                                                               h_src,
                                                               h_dst,
                                                               h_wgt,
                                                               num_edges,
                                                               store_transposed,
                                                               rocgraph_bool_false,
                                                               is_symmetric,
                                                               &p_graph,
                                                               &ret_error);

        if(h_seeds == nullptr)
        {
            CHECK_ROCGRAPH_SUCCESS_ERROR(
                rocgraph_select_random_vertices(
                    p_handle, p_graph, rng_state, num_vertices_to_sample, &seeds, &ret_error),
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

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_betweenness_centrality(p_handle,
                                                                     p_graph,
                                                                     seeds_view,
                                                                     normalized,
                                                                     include_endpoints,
                                                                     rocgraph_bool_false,
                                                                     &p_result,
                                                                     &ret_error),
                                     ret_error);

        rocgraph_type_erased_device_array_view_t* vertices;
        rocgraph_type_erased_device_array_view_t* centralities;

        vertices     = rocgraph_centrality_result_get_vertices(p_result);
        centralities = rocgraph_centrality_result_get_values(p_result);

        std::vector<vertex_t> h_vertices(num_vertices);
        std::vector<weight_t> h_centralities(num_vertices);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_vertices.data(), vertices, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_centralities.data(), centralities, &ret_error),
            ret_error);

        ROCGRAPH_CLIENTS_EXPECT_INDIRECT_ARRAY_NEAR_TOLERANCE(num_vertices,
                                                              h_result,
                                                              1,
                                                              h_vertices.data(),
                                                              h_centralities.data(),
                                                              1,
                                                              (const vertex_t*)nullptr,
                                                              0.0001);

        rocgraph_centrality_result_free(p_result);

        rocgraph_type_erased_device_array_view_free(seeds_view);
        rocgraph_type_erased_device_array_free(seeds);
        rocgraph_sg_graph_free(p_graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    template <typename weight_t, typename vertex_t>
    void generic_betweenness_centrality_bench(vertex_t*     h_src,
                                              vertex_t*     h_dst,
                                              weight_t*     h_wgt,
                                              vertex_t*     h_seeds,
                                              size_t        num_vertices,
                                              size_t        num_edges,
                                              size_t        num_seeds,
                                              rocgraph_bool store_transposed,
                                              rocgraph_bool is_symmetric,
                                              rocgraph_bool normalized,
                                              rocgraph_bool include_endpoints,
                                              size_t        num_vertices_to_sample)
    {
        rocgraph_error_t*                         ret_error;
        rocgraph_handle_t*                        p_handle   = nullptr;
        rocgraph_graph_t*                         p_graph    = nullptr;
        rocgraph_centrality_result_t*             p_result   = nullptr;
        rocgraph_rng_state_t*                     rng_state  = nullptr;
        rocgraph_type_erased_device_array_t*      seeds      = nullptr;
        rocgraph_type_erased_device_array_view_t* seeds_view = nullptr;

        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_rng_state_create(p_handle, 0, &rng_state, &ret_error),
                                     ret_error);
        rocgraph_clients_create_test_graph<weight_t, vertex_t>(p_handle,
                                                               h_src,
                                                               h_dst,
                                                               h_wgt,
                                                               num_edges,
                                                               store_transposed,
                                                               rocgraph_bool_false,
                                                               is_symmetric,
                                                               &p_graph,
                                                               &ret_error);

        if(h_seeds == nullptr)
        {
            CHECK_ROCGRAPH_SUCCESS_ERROR(
                rocgraph_select_random_vertices(
                    p_handle, p_graph, rng_state, num_vertices_to_sample, &seeds, &ret_error),
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

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_betweenness_centrality(p_handle,
                                                                     p_graph,
                                                                     seeds_view,
                                                                     normalized,
                                                                     include_endpoints,
                                                                     rocgraph_bool_false,
                                                                     &p_result,
                                                                     &ret_error),
                                     ret_error);

        rocgraph_centrality_result_free(p_result);

        rocgraph_type_erased_device_array_view_free(seeds_view);
        rocgraph_type_erased_device_array_free(seeds);
        rocgraph_sg_graph_free(p_graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    void BetweennessCentralityFull(const Arguments& arg)
    {
        using vertex_t = int32_t;
        using weight_t = float;

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
        weight_t h_result[] = {0, 3.66667, 0.833333, 2.16667, 0.833333, 0.5};

        generic_betweenness_centrality_test<weight_t, vertex_t>(h_src,
                                                                h_dst,
                                                                h_wgt,
                                                                nullptr,
                                                                h_result,
                                                                num_vertices,
                                                                num_edges,
                                                                0,
                                                                rocgraph_bool_false,
                                                                rocgraph_bool_true,
                                                                rocgraph_bool_false,
                                                                rocgraph_bool_false,
                                                                6);
    }

    void BetweennessCentralityFullDirected(const Arguments& arg)
    {
        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        weight_t h_result[] = {0, 4, 0, 2, 1, 0};

        generic_betweenness_centrality_test<weight_t, vertex_t>(h_src,
                                                                h_dst,
                                                                h_wgt,
                                                                nullptr,
                                                                h_result,
                                                                num_vertices,
                                                                num_edges,
                                                                0,
                                                                rocgraph_bool_false,
                                                                rocgraph_bool_false,
                                                                rocgraph_bool_false,
                                                                rocgraph_bool_false,
                                                                6);
    }

    void BetweennessCentralitySpecificNormalized(const Arguments& arg)
    {
        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 16;
        size_t num_vertices = 6;
        size_t num_seeds    = 2;

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
        vertex_t h_seeds[]  = {0, 3};
        weight_t h_result[] = {0, 0.475, 0.2, 0.1, 0.05, 0.075};

        generic_betweenness_centrality_test<weight_t, vertex_t>(h_src,
                                                                h_dst,
                                                                h_wgt,
                                                                h_seeds,
                                                                h_result,
                                                                num_vertices,
                                                                num_edges,
                                                                num_seeds,
                                                                rocgraph_bool_false,
                                                                rocgraph_bool_false,
                                                                rocgraph_bool_true,
                                                                rocgraph_bool_false,
                                                                num_seeds);
    }

    void BetweennessCentralitySpecificUnnormalized(const Arguments& arg)
    {
        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 16;
        size_t num_vertices = 6;
        size_t num_seeds    = 2;

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
        vertex_t h_seeds[]  = {0, 3};
        weight_t h_result[] = {0, 3.16667, 1.33333, 0.666667, 0.333333, 0.5};

        generic_betweenness_centrality_test<weight_t, vertex_t>(h_src,
                                                                h_dst,
                                                                h_wgt,
                                                                h_seeds,
                                                                h_result,
                                                                num_vertices,
                                                                num_edges,
                                                                num_seeds,
                                                                rocgraph_bool_false,
                                                                rocgraph_bool_false,
                                                                rocgraph_bool_false,
                                                                rocgraph_bool_false,
                                                                num_seeds);
    }

    void BetweennessCentralityTestEndpoints(const Arguments& arg)
    {
        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        weight_t h_result[] = {0.166667, 0.3, 0.166667, 0.2, 0.166667, 0.166667};

        generic_betweenness_centrality_test<weight_t, vertex_t>(h_src,
                                                                h_dst,
                                                                h_wgt,
                                                                nullptr,
                                                                h_result,
                                                                num_vertices,
                                                                num_edges,
                                                                0,
                                                                rocgraph_bool_false,
                                                                rocgraph_bool_false,
                                                                rocgraph_bool_true,
                                                                rocgraph_bool_true,
                                                                6);
    }

    void BetweennessCentralityFullDirectedNormalizedKarate(const Arguments& arg)
    {
        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 156;
        size_t num_vertices = 34;

        vertex_t h_src[]
            = {1,  2,  3,  4,  5,  6,  7,  8,  10, 11, 12, 13, 17, 19, 21, 31, 2,  3,  7,  13,
               17, 19, 21, 30, 3,  7,  8,  9,  13, 27, 28, 32, 7,  12, 13, 6,  10, 6,  10, 16,
               16, 30, 32, 33, 33, 33, 32, 33, 32, 33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29,
               32, 33, 25, 27, 31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33, 0,  0,
               0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,
               1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  4,  4,  5,  5,  5,  6,  8,
               8,  8,  9,  13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22, 23, 23, 23, 23, 23,
               24, 24, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32};

        vertex_t h_dst[]
            = {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,
               1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  4,  4,  5,  5,  5,
               6,  8,  8,  8,  9,  13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22, 23, 23, 23,
               23, 23, 24, 24, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 1,  2,
               3,  4,  5,  6,  7,  8,  10, 11, 12, 13, 17, 19, 21, 31, 2,  3,  7,  13, 17, 19,
               21, 30, 3,  7,  8,  9,  13, 27, 28, 32, 7,  12, 13, 6,  10, 6,  10, 16, 16, 30,
               32, 33, 33, 33, 32, 33, 32, 33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29, 32, 33,
               25, 27, 31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33};

        weight_t h_wgt[]
            = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

        weight_t h_result[]
            = {462.142914, 56.957146, 151.701584, 12.576191,  0.666667,   31.666668, 31.666668,
               0.000000,   59.058739, 0.895238,   0.666667,   0.000000,   0.000000,  48.431747,
               0.000000,   0.000000,  0.000000,   0.000000,   0.000000,   34.293652, 0.000000,
               0.000000,   0.000000,  18.600000,  2.333333,   4.055556,   0.000000,  23.584126,
               1.895238,   3.085714,  15.219049,  146.019043, 153.380981, 321.103180};

        generic_betweenness_centrality_test<weight_t, vertex_t>(h_src,
                                                                h_dst,
                                                                h_wgt,
                                                                nullptr,
                                                                h_result,
                                                                num_vertices,
                                                                num_edges,
                                                                0,
                                                                rocgraph_bool_false,
                                                                rocgraph_bool_false,
                                                                rocgraph_bool_false,
                                                                rocgraph_bool_false,
                                                                34);
    }

} // namespace

template <typename I, typename T>
void testing_rocgraph_betweenness_centrality_bad_arg(const Arguments& arg)
{
}

template <typename I, typename T>
void testing_rocgraph_betweenness_centrality(const Arguments& arg)
{

    using vertex_t            = I;
    using weight_t            = T;
    const size_t num_vertices = arg.M;
    const size_t num_edges    = (arg.M - 1);

    std::vector<vertex_t> h_src(num_edges);
    std::vector<vertex_t> h_dst(num_edges);
    std::vector<weight_t> h_wgt(num_edges, 1);

    for(size_t i = 0; i < num_edges; ++i)
    {
        h_src[i] = 0;
        h_dst[i] = i + 1;
    }

    const rocgraph_bool store_transposed  = rocgraph_bool_false;
    const rocgraph_bool is_symmetric      = rocgraph_bool_false;
    const rocgraph_bool normalized        = rocgraph_bool_false;
    const rocgraph_bool include_endpoints = rocgraph_bool_false;

    generic_betweenness_centrality_bench<weight_t, vertex_t>(h_src.data(),
                                                             h_dst.data(),
                                                             h_wgt.data(),
                                                             nullptr,
                                                             num_vertices,
                                                             num_edges,
                                                             0,
                                                             store_transposed,
                                                             is_symmetric,
                                                             normalized,
                                                             include_endpoints,
                                                             num_vertices);

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

#define INSTANTIATE(I, T)                                                                      \
    template void testing_rocgraph_betweenness_centrality_bad_arg<I, T>(const Arguments& arg); \
    template void testing_rocgraph_betweenness_centrality<I, T>(const Arguments& arg)

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);

void testing_rocgraph_betweenness_centrality_extra(const Arguments& arg)
{
    testing_dispatch_extra(arg,
                           "BetweennessCentralityFull",
                           BetweennessCentralityFull,
                           "BetweennessCentralityFullDirected",
                           BetweennessCentralityFullDirected,
                           "BetweennessCentralitySpecificNormalized",
                           BetweennessCentralitySpecificNormalized,
                           "BetweennessCentralitySpecificUnnormalized",
                           BetweennessCentralitySpecificUnnormalized,
                           "BetweennessCentralityTestEndpoints",
                           BetweennessCentralityTestEndpoints,
                           "BetweennessCentralityFullDirectedNormalizedKarate",
                           BetweennessCentralityFullDirectedNormalizedKarate);
}
