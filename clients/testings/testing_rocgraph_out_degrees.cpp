// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/*
 * Copyright (C) 2024, NVIDIA CORPORATION.
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

#include "rocgraph_clients_expect_eq.hpp"

#include "rocgraph/rocgraph.h"
#include "testing.hpp"
#include "testing_rocgraph_out_degrees.hpp"

/*
 * Simple check of creating a graph from a COO on device memory.
 */
namespace
{
    template <typename weight_t, typename edge_t, typename vertex_t>
    void generic_degrees_test(vertex_t*     h_src,
                              vertex_t*     h_dst,
                              weight_t*     h_wgt,
                              size_t        num_vertices,
                              size_t        num_edges,
                              vertex_t*     h_vertices,
                              size_t        num_vertices_to_compute,
                              rocgraph_bool in_degrees,
                              rocgraph_bool out_degrees,
                              rocgraph_bool store_transposed,
                              rocgraph_bool is_symmetric,
                              edge_t*       h_in_degrees,
                              edge_t*       h_out_degrees)
    {
        rocgraph_error_t* ret_error;

        rocgraph_handle_t*         p_handle = nullptr;
        rocgraph_graph_t*          graph    = nullptr;
        rocgraph_degrees_result_t* result   = nullptr;

        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        rocgraph_clients_create_test_graph(p_handle,
                                           h_src,
                                           h_dst,
                                           h_wgt,
                                           num_edges,
                                           store_transposed,
                                           rocgraph_bool_false,
                                           is_symmetric,
                                           &graph,
                                           &ret_error);

        if(h_vertices == nullptr)
        {
            if(in_degrees && out_degrees)
            {
                CHECK_ROCGRAPH_SUCCESS_ERROR(
                    rocgraph_degrees(
                        p_handle, graph, nullptr, rocgraph_bool_false, &result, &ret_error),
                    ret_error);
            }
            else if(in_degrees)
            {
                CHECK_ROCGRAPH_SUCCESS_ERROR(
                    rocgraph_in_degrees(
                        p_handle, graph, nullptr, rocgraph_bool_false, &result, &ret_error),
                    ret_error);
            }
            else
            {
                CHECK_ROCGRAPH_SUCCESS_ERROR(
                    rocgraph_out_degrees(
                        p_handle, graph, nullptr, rocgraph_bool_false, &result, &ret_error),
                    ret_error);
            }
        }
        else
        {
            rocgraph_type_erased_device_array_t*      vertices      = nullptr;
            rocgraph_type_erased_device_array_view_t* vertices_view = nullptr;

            CHECK_ROCGRAPH_SUCCESS_ERROR(
                rocgraph_type_erased_device_array_create(p_handle,
                                                         num_vertices_to_compute,
                                                         rocgraph_data_type_id_int32,
                                                         &vertices,
                                                         &ret_error),
                ret_error);

            vertices_view = rocgraph_type_erased_device_array_view(vertices);

            CHECK_ROCGRAPH_SUCCESS_ERROR(
                rocgraph_type_erased_device_array_view_copy_from_host(
                    p_handle, vertices_view, (rocgraph_byte_t*)h_vertices, &ret_error),
                ret_error);
            if(in_degrees && out_degrees)
            {
                CHECK_ROCGRAPH_SUCCESS_ERROR(
                    rocgraph_degrees(
                        p_handle, graph, vertices_view, rocgraph_bool_false, &result, &ret_error),
                    ret_error);
            }
            else if(in_degrees)
            {
                CHECK_ROCGRAPH_SUCCESS_ERROR(
                    rocgraph_in_degrees(
                        p_handle, graph, vertices_view, rocgraph_bool_false, &result, &ret_error),
                    ret_error);
            }
            else
            {
                CHECK_ROCGRAPH_SUCCESS_ERROR(
                    rocgraph_out_degrees(
                        p_handle, graph, vertices_view, rocgraph_bool_false, &result, &ret_error),
                    ret_error);
            }
        }

        rocgraph_type_erased_device_array_view_t* result_vertices;
        rocgraph_type_erased_device_array_view_t* result_in_degrees;
        rocgraph_type_erased_device_array_view_t* result_out_degrees;

        result_vertices    = rocgraph_degrees_result_get_vertices(result);
        result_in_degrees  = rocgraph_degrees_result_get_in_degrees(result);
        result_out_degrees = rocgraph_degrees_result_get_out_degrees(result);

        size_t num_result_vertices = rocgraph_type_erased_device_array_view_size(result_vertices);

        std::vector<vertex_t> h_result_vertices(num_result_vertices);
        std::vector<edge_t>   h_result_in_degrees(num_result_vertices);
        std::vector<edge_t>   h_result_out_degrees(num_result_vertices);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_vertices.data(), result_vertices, &ret_error),
            ret_error);
        if(result_in_degrees != nullptr)
        {
            CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_to_host(
                                             p_handle,
                                             (rocgraph_byte_t*)h_result_in_degrees.data(),
                                             result_in_degrees,
                                             &ret_error),
                                         ret_error);
        }

        if(result_out_degrees != nullptr)
        {
            CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_to_host(
                                             p_handle,
                                             (rocgraph_byte_t*)h_result_out_degrees.data(),
                                             result_out_degrees,
                                             &ret_error),
                                         ret_error);
        }

        if(h_vertices != nullptr)
        {
            ROCGRAPH_CLIENTS_EXPECT_EQ(num_result_vertices, num_vertices_to_compute);
        }
        else
        {
            ROCGRAPH_CLIENTS_EXPECT_EQ(num_result_vertices, num_vertices);
        }

        if(h_in_degrees != nullptr)
        {
            ROCGRAPH_CLIENTS_EXPECT_INDIRECT_ARRAY_EQ(num_result_vertices,
                                                      (const edge_t*)h_result_in_degrees.data(),
                                                      1,
                                                      (const vertex_t*)nullptr,
                                                      (const edge_t*)h_in_degrees,
                                                      1,
                                                      (const vertex_t*)h_result_vertices.data());
        }

        if(h_out_degrees != nullptr)
        {
            ROCGRAPH_CLIENTS_EXPECT_INDIRECT_ARRAY_EQ(num_result_vertices,
                                                      (const edge_t*)h_result_out_degrees.data(),
                                                      1,
                                                      (const vertex_t*)nullptr,
                                                      (const edge_t*)h_out_degrees,
                                                      1,
                                                      (const vertex_t*)h_result_vertices.data());
        }

        rocgraph_degrees_result_free(result);
        rocgraph_graph_free(graph);
        rocgraph_error_free(ret_error);
    }

    void Degrees(const Arguments& arg)
    {
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        using vertex_t           = int32_t;
        using edge_t             = int32_t;
        using weight_t           = float;
        vertex_t h_src[]         = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]         = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]         = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t h_in_degrees[]  = {1, 2, 0, 2, 1, 2};
        vertex_t h_out_degrees[] = {1, 2, 3, 1, 1, 0};

        generic_degrees_test<weight_t, edge_t, vertex_t>(h_src,
                                                         h_dst,
                                                         h_wgt,
                                                         num_vertices,
                                                         num_edges,
                                                         nullptr,
                                                         0,
                                                         rocgraph_bool_true,
                                                         rocgraph_bool_true,
                                                         rocgraph_bool_false,
                                                         rocgraph_bool_false,
                                                         h_in_degrees,
                                                         h_out_degrees);
    }

    void DegreesSymmetric(const Arguments& arg)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 6;
        using vertex_t      = int32_t;
        using edge_t        = int32_t;
        using weight_t      = float;

        vertex_t h_src[]         = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_dst[]         = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
        weight_t h_wgt[]         = {0.1f,
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
        vertex_t h_in_degrees[]  = {2, 4, 3, 3, 2, 2};
        vertex_t h_out_degrees[] = {2, 4, 3, 3, 2, 2};

        generic_degrees_test<weight_t, edge_t, vertex_t>(h_src,
                                                         h_dst,
                                                         h_wgt,
                                                         num_vertices,
                                                         num_edges,
                                                         nullptr,
                                                         0,
                                                         rocgraph_bool_true,
                                                         rocgraph_bool_true,
                                                         rocgraph_bool_false,
                                                         rocgraph_bool_true,
                                                         h_in_degrees,
                                                         h_out_degrees);
    }

    void InDegrees(const Arguments& arg)
    {
        size_t num_edges    = 8;
        size_t num_vertices = 6;
        using vertex_t      = int32_t;
        using edge_t        = int32_t;
        using weight_t      = float;

        vertex_t h_src[]        = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]        = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]        = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t h_in_degrees[] = {1, 2, 0, 2, 1, 2};

        generic_degrees_test<weight_t, edge_t, vertex_t>(h_src,
                                                         h_dst,
                                                         h_wgt,
                                                         num_vertices,
                                                         num_edges,
                                                         nullptr,
                                                         0,
                                                         rocgraph_bool_true,
                                                         rocgraph_bool_false,
                                                         rocgraph_bool_false,
                                                         rocgraph_bool_true,
                                                         h_in_degrees,
                                                         nullptr);
    }

    void OutDegrees(const Arguments& arg)
    {
        size_t num_edges    = 8;
        size_t num_vertices = 6;
        using vertex_t      = int32_t;
        using edge_t        = int32_t;
        using weight_t      = float;

        vertex_t h_src[]         = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]         = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]         = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t h_out_degrees[] = {1, 2, 3, 1, 1, 0};

        generic_degrees_test<weight_t, edge_t, vertex_t>(h_src,
                                                         h_dst,
                                                         h_wgt,
                                                         num_vertices,
                                                         num_edges,
                                                         nullptr,
                                                         0,
                                                         rocgraph_bool_false,
                                                         rocgraph_bool_true,
                                                         rocgraph_bool_false,
                                                         rocgraph_bool_true,
                                                         nullptr,
                                                         h_out_degrees);
    }

    void DegreesSubset(const Arguments& arg)
    {
        size_t num_edges               = 8;
        size_t num_vertices            = 6;
        size_t num_vertices_to_compute = 3;
        using vertex_t                 = int32_t;
        using edge_t                   = int32_t;
        using weight_t                 = float;

        vertex_t h_src[]         = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]         = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]         = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t h_vertices[]    = {2, 3, 5};
        vertex_t h_in_degrees[]  = {-1, -1, 0, 2, -1, 2};
        vertex_t h_out_degrees[] = {-1, -1, 3, 1, -1, 0};

        generic_degrees_test<weight_t, edge_t, vertex_t>(h_src,
                                                         h_dst,
                                                         h_wgt,
                                                         num_vertices,
                                                         num_edges,
                                                         h_vertices,
                                                         num_vertices_to_compute,
                                                         rocgraph_bool_true,
                                                         rocgraph_bool_true,
                                                         rocgraph_bool_false,
                                                         rocgraph_bool_false,
                                                         h_in_degrees,
                                                         h_out_degrees);
    }

    void DegreesSymmetricSubset(const Arguments& arg)
    {
        size_t num_edges               = 16;
        size_t num_vertices            = 6;
        size_t num_vertices_to_compute = 3;

        using vertex_t = int32_t;
        using edge_t   = int32_t;
        using weight_t = float;

        vertex_t h_src[]         = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_dst[]         = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
        weight_t h_wgt[]         = {0.1f,
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
        vertex_t h_vertices[]    = {2, 3, 5};
        vertex_t h_in_degrees[]  = {-1, -1, 3, 3, -1, 2};
        vertex_t h_out_degrees[] = {-1, -1, 3, 3, -1, 2};

        generic_degrees_test<weight_t, edge_t, vertex_t>(h_src,
                                                         h_dst,
                                                         h_wgt,
                                                         num_vertices,
                                                         num_edges,
                                                         h_vertices,
                                                         num_vertices_to_compute,
                                                         rocgraph_bool_true,
                                                         rocgraph_bool_true,
                                                         rocgraph_bool_false,
                                                         rocgraph_bool_true,
                                                         h_in_degrees,
                                                         h_out_degrees);
    }

    void InDegreesSubset(const Arguments& arg)
    {
        size_t num_edges               = 8;
        size_t num_vertices            = 6;
        size_t num_vertices_to_compute = 3;
        using vertex_t                 = int32_t;
        using edge_t                   = int32_t;
        using weight_t                 = float;

        vertex_t h_src[]        = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]        = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]        = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t h_vertices[]   = {2, 3, 5};
        vertex_t h_in_degrees[] = {-1, -1, 0, 2, -1, 2};

        generic_degrees_test<weight_t, edge_t, vertex_t>(h_src,
                                                         h_dst,
                                                         h_wgt,
                                                         num_vertices,
                                                         num_edges,
                                                         h_vertices,
                                                         num_vertices_to_compute,
                                                         rocgraph_bool_true,
                                                         rocgraph_bool_false,
                                                         rocgraph_bool_false,
                                                         rocgraph_bool_true,
                                                         h_in_degrees,
                                                         nullptr);
    }

    void OutDegreesSubset(const Arguments& arg)
    {
        size_t num_edges               = 8;
        size_t num_vertices            = 6;
        size_t num_vertices_to_compute = 3;
        using vertex_t                 = int32_t;
        using edge_t                   = int32_t;
        using weight_t                 = float;

        vertex_t h_src[]         = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]         = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]         = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t h_vertices[]    = {2, 3, 5};
        vertex_t h_out_degrees[] = {-1, -1, 3, 1, -1, 0};

        generic_degrees_test<weight_t, edge_t, vertex_t>(h_src,
                                                         h_dst,
                                                         h_wgt,
                                                         num_vertices,
                                                         num_edges,
                                                         h_vertices,
                                                         num_vertices_to_compute,
                                                         rocgraph_bool_false,
                                                         rocgraph_bool_true,
                                                         rocgraph_bool_false,
                                                         rocgraph_bool_true,
                                                         nullptr,
                                                         h_out_degrees);
    }

} // namespace

template <typename T>
void testing_rocgraph_out_degrees_bad_arg(const Arguments& arg)
{
#ifdef TO_FILL

    const rocgraph_handle_t*                        handle{};
    rocgraph_graph_t*                               graph{};
    const rocgraph_type_erased_device_array_view_t* source_vertices{};
    rocgraph_bool                                   do_expensive_check{};
    rocgraph_degrees_result_t**                     result{};
    rocgraph_error_t**                              error{};
    auto                                            ret
        = rocgraph_out_degrees(handle, graph, source_vertices, do_expensive_check, result, error);
    if(ret != rocgraph_status_success)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_internal_error);
    }
    CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
#endif
}

template <typename T>
void testing_rocgraph_out_degrees(const Arguments& arg)
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

#define INSTANTIATE(TYPE)                                                           \
    template void testing_rocgraph_out_degrees_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_rocgraph_out_degrees<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_out_degrees_extra(const Arguments& arg)
{
    testing_dispatch_extra(arg,
                           "Degrees",
                           Degrees,
                           "DegreesSymmetric",
                           DegreesSymmetric,
                           "InDegrees",
                           InDegrees,
                           "OutDegrees",
                           OutDegrees,
                           "DegreesSubset",
                           DegreesSubset,
                           "DegreesSymmetricSubset",
                           DegreesSymmetricSubset,
                           "DegreesSymmetricSubset",
                           DegreesSymmetricSubset,
                           "InDegreesSubset",
                           InDegreesSubset,
                           "OutDegreesSubset",
                           OutDegreesSubset);
}
