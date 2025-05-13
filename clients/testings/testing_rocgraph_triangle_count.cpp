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

#include "rocgraph_clients_create_test_graph.hpp"
#include "rocgraph_test.hpp"

#include "rocgraph/rocgraph.h"
#include "rocgraph_clients_skip_test.hpp"
#include "testing.hpp"
#include "testing_rocgraph_triangle_count.hpp"

namespace
{
    template <typename weight_t, typename edge_t, typename vertex_t>
    void generic_triangle_count_test(vertex_t*     h_src,
                                     vertex_t*     h_dst,
                                     weight_t*     h_wgt,
                                     vertex_t*     h_verts,
                                     edge_t*       h_result,
                                     size_t        num_vertices,
                                     size_t        num_edges,
                                     size_t        num_results,
                                     rocgraph_bool store_transposed)
    {

        rocgraph_error_t* ret_error;

        rocgraph_handle_t*                        p_handle     = nullptr;
        rocgraph_graph_t*                         p_graph      = nullptr;
        rocgraph_triangle_count_result_t*         p_result     = nullptr;
        rocgraph_type_erased_device_array_t*      p_start      = nullptr;
        rocgraph_type_erased_device_array_view_t* p_start_view = nullptr;

        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        rocgraph_clients_create_test_graph(p_handle,
                                           h_src,
                                           h_dst,
                                           h_wgt,
                                           num_edges,
                                           store_transposed,
                                           rocgraph_bool_false,
                                           rocgraph_bool_true,
                                           &p_graph,
                                           &ret_error);

        if(h_verts != nullptr)
        {
            CHECK_ROCGRAPH_SUCCESS_ERROR(
                rocgraph_type_erased_device_array_create(
                    p_handle, num_results, rocgraph_data_type_id_int32, &p_start, &ret_error),
                ret_error);

            p_start_view = rocgraph_type_erased_device_array_view(p_start);

            CHECK_ROCGRAPH_SUCCESS_ERROR(
                rocgraph_type_erased_device_array_view_copy_from_host(
                    p_handle, p_start_view, (rocgraph_byte_t*)h_verts, &ret_error),
                ret_error);
        }

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_triangle_count(
                p_handle, p_graph, p_start_view, rocgraph_bool_false, &p_result, &ret_error),
            ret_error);

        rocgraph_type_erased_device_array_view_t* vertices;
        rocgraph_type_erased_device_array_view_t* counts;

        vertices = rocgraph_triangle_count_result_get_vertices(p_result);
        counts   = rocgraph_triangle_count_result_get_counts(p_result);

        vertex_t num_local_results = rocgraph_type_erased_device_array_view_size(vertices);

        std::vector<vertex_t> h_vertices(num_local_results);
        std::vector<edge_t>   h_counts(num_local_results);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_vertices.data(), vertices, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_counts.data(), counts, &ret_error),
            ret_error);

        ROCGRAPH_CLIENTS_EXPECT_SEGMENT_EQ(num_local_results, h_result, h_counts.data());

        rocgraph_triangle_count_result_free(p_result);

        rocgraph_sg_graph_free(p_graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    void TriangleCount(const Arguments& arg)
    {

        using vertex_t      = int32_t;
        using edge_t        = int32_t;
        using weight_t      = float;
        size_t num_edges    = 16;
        size_t num_vertices = 6;
        size_t num_results  = 4;

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
        vertex_t h_verts[]  = {0, 1, 2, 4};
        edge_t   h_result[] = {1, 2, 2, 0};

        // Triangle Count wants store_transposed = rocgraph_bool_false
        generic_triangle_count_test(h_src,
                                    h_dst,
                                    h_wgt,
                                    h_verts,
                                    h_result,
                                    num_vertices,
                                    num_edges,
                                    num_results,
                                    rocgraph_bool_false);
    }

    void TriangleCountDolphins(const Arguments& arg)
    {
        using vertex_t = int32_t;
        using edge_t   = int32_t;
        using weight_t = float;

        size_t num_edges    = 318;
        size_t num_vertices = 62;

        vertex_t h_src[] = {
            10, 14, 15, 40, 42, 47, 17, 19, 26, 27, 28, 36, 41, 54, 10, 42, 44, 61, 8,  14, 59, 51,
            9,  13, 56, 57, 9,  13, 17, 54, 56, 57, 19, 27, 30, 40, 54, 20, 28, 37, 45, 59, 13, 17,
            32, 41, 57, 29, 42, 47, 51, 33, 17, 32, 41, 54, 57, 16, 24, 33, 34, 37, 38, 40, 43, 50,
            52, 18, 24, 40, 45, 55, 59, 20, 33, 37, 38, 50, 22, 25, 27, 31, 57, 20, 21, 24, 29, 45,
            51, 30, 54, 28, 36, 38, 44, 47, 50, 29, 33, 37, 45, 51, 36, 45, 51, 29, 45, 51, 26, 27,
            27, 30, 47, 35, 43, 45, 51, 52, 42, 47, 60, 34, 37, 38, 40, 43, 50, 37, 44, 49, 37, 39,
            40, 59, 40, 43, 45, 61, 43, 44, 52, 58, 57, 52, 54, 57, 47, 50, 46, 53, 50, 51, 59, 49,
            57, 51, 55, 61, 57, 0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,
            2,  3,  3,  3,  4,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  8,  8,
            8,  8,  8,  9,  9,  9,  9,  9,  10, 10, 10, 11, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14,
            14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17,
            18, 18, 18, 18, 18, 18, 19, 19, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 23, 23, 23,
            24, 24, 24, 25, 25, 26, 28, 28, 29, 29, 29, 29, 29, 30, 30, 32, 33, 33, 33, 33, 33, 33,
            34, 34, 34, 36, 36, 36, 36, 37, 37, 37, 37, 38, 38, 38, 38, 39, 40, 41, 41, 42, 42, 43,
            43, 45, 45, 45, 46, 48, 50, 51, 53, 54};

        vertex_t h_dst[] = {
            0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  4,
            5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  9,  9,
            9,  9,  9,  10, 10, 10, 11, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14,
            14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18,
            18, 19, 19, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 23, 23, 23, 24, 24, 24, 25, 25,
            26, 28, 28, 29, 29, 29, 29, 29, 30, 30, 32, 33, 33, 33, 33, 33, 33, 34, 34, 34, 36, 36,
            36, 36, 37, 37, 37, 37, 38, 38, 38, 38, 39, 40, 41, 41, 42, 42, 43, 43, 45, 45, 45, 46,
            48, 50, 51, 53, 54, 10, 14, 15, 40, 42, 47, 17, 19, 26, 27, 28, 36, 41, 54, 10, 42, 44,
            61, 8,  14, 59, 51, 9,  13, 56, 57, 9,  13, 17, 54, 56, 57, 19, 27, 30, 40, 54, 20, 28,
            37, 45, 59, 13, 17, 32, 41, 57, 29, 42, 47, 51, 33, 17, 32, 41, 54, 57, 16, 24, 33, 34,
            37, 38, 40, 43, 50, 52, 18, 24, 40, 45, 55, 59, 20, 33, 37, 38, 50, 22, 25, 27, 31, 57,
            20, 21, 24, 29, 45, 51, 30, 54, 28, 36, 38, 44, 47, 50, 29, 33, 37, 45, 51, 36, 45, 51,
            29, 45, 51, 26, 27, 27, 30, 47, 35, 43, 45, 51, 52, 42, 47, 60, 34, 37, 38, 40, 43, 50,
            37, 44, 49, 37, 39, 40, 59, 40, 43, 45, 61, 43, 44, 52, 58, 57, 52, 54, 57, 47, 50, 46,
            53, 50, 51, 59, 49, 57, 51, 55, 61, 57};

        weight_t h_wgt[]
            = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

        vertex_t h_verts[]   = {11, 48, 0};
        edge_t   h_result[]  = {0, 0, 5};
        size_t   num_results = 3;

        // Triangle Count wants store_transposed = rocgraph_bool_false
        generic_triangle_count_test(h_src,
                                    h_dst,
                                    h_wgt,
                                    h_verts,
                                    h_result,
                                    num_vertices,
                                    num_edges,
                                    num_results,
                                    rocgraph_bool_false);
    }

} // namespace

template <typename T>
void testing_rocgraph_triangle_count_bad_arg(const Arguments& arg)
{
#ifdef TO_FILL

    const rocgraph_handle_t*                        handle{};
    rocgraph_graph_t*                               graph{};
    const rocgraph_type_erased_device_array_view_t* start{};
    rocgraph_bool                                   do_expensive_check{};
    rocgraph_triangle_count_result_t**              result{};
    rocgraph_error_t**                              error{};
    auto ret = rocgraph_triangle_count(handle, graph, start, do_expensive_check, result, error);
    if(ret != rocgraph_status_success)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_internal_error);
    }
    CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);

#endif
}

template <typename T>
void testing_rocgraph_triangle_count(const Arguments& arg)
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

#define INSTANTIATE(TYPE)                                                              \
    template void testing_rocgraph_triangle_count_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_rocgraph_triangle_count<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_triangle_count_extra(const Arguments& arg)
{
    testing_dispatch_extra(
        arg, "TriangleCount", TriangleCount, "TriangleCountDolphins", TriangleCountDolphins);
}
