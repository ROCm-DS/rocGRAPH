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

#include "testing_rocgraph_hits.hpp"
#include "rocgraph/rocgraph.h"
#include "rocgraph_clients_create_test_graph.hpp"
#include "rocgraph_test.hpp"
#include "testing.hpp"

namespace
{

    template <typename weight_t, typename vertex_t>
    void generic_hits_test(vertex_t*     h_src,
                           vertex_t*     h_dst,
                           weight_t*     h_wgt,
                           size_t        num_vertices,
                           size_t        num_edges,
                           vertex_t*     h_initial_vertices,
                           weight_t*     h_initial_hubs,
                           size_t        num_initial_vertices,
                           weight_t*     h_result_hubs,
                           weight_t*     h_result_authorities,
                           rocgraph_bool store_transposed,
                           rocgraph_bool renumber,
                           rocgraph_bool normalize,
                           double        epsilon,
                           size_t        max_iterations)
    {

        rocgraph_error_t* ret_error;

        rocgraph_handle_t*      p_handle = nullptr;
        rocgraph_graph_t*       p_graph  = nullptr;
        rocgraph_hits_result_t* p_result = nullptr;

        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        rocgraph_clients_create_test_graph(p_handle,
                                           h_src,
                                           h_dst,
                                           h_wgt,
                                           num_edges,
                                           store_transposed,
                                           renumber,
                                           rocgraph_bool_false,
                                           &p_graph,
                                           &ret_error);

        if(h_initial_vertices == nullptr)
        {
            CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_hits(p_handle,
                                                       p_graph,
                                                       epsilon,
                                                       max_iterations,
                                                       nullptr,
                                                       nullptr,
                                                       normalize,
                                                       rocgraph_bool_false,
                                                       &p_result,
                                                       &ret_error),
                                         ret_error);
        }
        else
        {
            rocgraph_type_erased_device_array_t*      initial_vertices;
            rocgraph_type_erased_device_array_t*      initial_hubs;
            rocgraph_type_erased_device_array_view_t* initial_vertices_view;
            rocgraph_type_erased_device_array_view_t* initial_hubs_view;

            CHECK_ROCGRAPH_SUCCESS_ERROR(
                rocgraph_type_erased_device_array_create(p_handle,
                                                         num_initial_vertices,
                                                         rocgraph_data_type_id_int32,
                                                         &initial_vertices,
                                                         &ret_error),
                ret_error);

            CHECK_ROCGRAPH_SUCCESS_ERROR(
                rocgraph_type_erased_device_array_create(p_handle,
                                                         num_initial_vertices,
                                                         rocgraph_data_type_id_float32,
                                                         &initial_hubs,
                                                         &ret_error),
                ret_error);

            initial_vertices_view = rocgraph_type_erased_device_array_view(initial_vertices);
            initial_hubs_view     = rocgraph_type_erased_device_array_view(initial_hubs);

            CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                             p_handle,
                                             initial_vertices_view,
                                             (rocgraph_byte_t*)h_initial_vertices,
                                             &ret_error),
                                         ret_error);

            CHECK_ROCGRAPH_SUCCESS_ERROR(
                rocgraph_type_erased_device_array_view_copy_from_host(
                    p_handle, initial_hubs_view, (rocgraph_byte_t*)h_initial_hubs, &ret_error),
                ret_error);

            CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_hits(p_handle,
                                                       p_graph,
                                                       epsilon,
                                                       max_iterations,
                                                       initial_vertices_view,
                                                       initial_hubs_view,
                                                       normalize,
                                                       rocgraph_bool_false,
                                                       &p_result,
                                                       &ret_error),
                                         ret_error);
        }

        rocgraph_type_erased_device_array_view_t* vertices;
        rocgraph_type_erased_device_array_view_t* hubs;
        rocgraph_type_erased_device_array_view_t* authorities;

        vertices    = rocgraph_hits_result_get_vertices(p_result);
        hubs        = rocgraph_hits_result_get_hubs(p_result);
        authorities = rocgraph_hits_result_get_authorities(p_result);

        std::vector<vertex_t> h_vertices(num_vertices);
        std::vector<weight_t> h_hubs(num_vertices);
        std::vector<weight_t> h_authorities(num_vertices);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_vertices.data(), vertices, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_hubs.data(), hubs, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_authorities.data(), authorities, &ret_error),
            ret_error);

        ROCGRAPH_CLIENTS_EXPECT_INDIRECT_ARRAY_NEAR_TOLERANCE(num_vertices,
                                                              h_result_hubs,
                                                              1,
                                                              h_vertices.data(),
                                                              h_hubs.data(),
                                                              1,
                                                              (const vertex_t*)nullptr,
                                                              0.001);

        ROCGRAPH_CLIENTS_EXPECT_INDIRECT_ARRAY_NEAR_TOLERANCE(num_vertices,
                                                              h_result_authorities,
                                                              1,
                                                              h_vertices.data(),
                                                              h_authorities.data(),
                                                              1,
                                                              (const vertex_t*)nullptr,
                                                              0.001);

        rocgraph_hits_result_free(p_result);
        rocgraph_sg_graph_free(p_graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    void Hits(const Arguments& arg)
    {
        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t h_src[]         = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]         = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]         = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        weight_t h_hubs[]        = {0.347296, 0.532089, 1, 0.00003608, 0.00003608, 0};
        weight_t h_authorities[] = {0.652703, 0.879385, 0, 1, 0.347296, 0.00009136};

        double epsilon        = 0.00002;
        size_t max_iterations = 20;

        // hits wants store_transposed = rocgraph_bool_true
        generic_hits_test<weight_t, vertex_t>(h_src,
                                              h_dst,
                                              h_wgt,
                                              num_vertices,
                                              num_edges,
                                              nullptr,
                                              nullptr,
                                              0,
                                              h_hubs,
                                              h_authorities,
                                              rocgraph_bool_true,
                                              rocgraph_bool_false,
                                              rocgraph_bool_false,
                                              epsilon,
                                              max_iterations);
    }

    void HitsWithTranspose(const Arguments& arg)
    {

        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t h_src[]         = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]         = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]         = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        weight_t h_hubs[]        = {0.347296, 0.532089, 1, 0.00003608, 0.00003608, 0};
        weight_t h_authorities[] = {0.652703, 0.879385, 0, 1, 0.347296, 0.00009136};

        double epsilon        = 0.00002;
        size_t max_iterations = 20;

        // Hits wants store_transposed = rocgraph_bool_true
        //    This call will force rocgraph_hits to transpose the graph
        //    But we're passing src/dst backwards so the results will be the same
        generic_hits_test<weight_t, vertex_t>(h_src,
                                              h_dst,
                                              h_wgt,
                                              num_vertices,
                                              num_edges,
                                              nullptr,
                                              nullptr,
                                              0,
                                              h_hubs,
                                              h_authorities,
                                              rocgraph_bool_false,
                                              rocgraph_bool_false,
                                              rocgraph_bool_false,
                                              epsilon,
                                              max_iterations);
    }

    void HitsWithInitial(const Arguments& arg)
    {
        using vertex_t          = int32_t;
        using weight_t          = float;
        size_t num_edges        = 8;
        size_t num_vertices     = 6;
        size_t num_initial_hubs = 5;

        vertex_t h_src[]              = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]              = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]              = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        weight_t h_hubs[]             = {0.347296, 0.532089, 1, 0.00000959, 0.00000959, 0};
        weight_t h_authorities[]      = {0.652704, 0.879385, 0, 1, 0.347296, 0.00002428};
        vertex_t h_initial_vertices[] = {0, 1, 2, 3, 4};
        weight_t h_initial_hubs[]     = {0.347296, 0.532089, 1, 0.00003608, 0.00003608};

        double epsilon        = 0.00002;
        size_t max_iterations = 20;

        generic_hits_test<weight_t, vertex_t>(h_src,
                                              h_dst,
                                              h_wgt,
                                              num_vertices,
                                              num_edges,
                                              h_initial_vertices,
                                              h_initial_hubs,
                                              num_initial_hubs,
                                              h_hubs,
                                              h_authorities,
                                              rocgraph_bool_false,
                                              rocgraph_bool_false,
                                              rocgraph_bool_false,
                                              epsilon,
                                              max_iterations);
    }

    void HitsBigger(const Arguments& arg)
    {

        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 48;
        size_t num_vertices = 54;

        vertex_t h_src[] = {29, 45, 6,  8,  16, 45, 8,  16, 6,  38, 45, 45, 48, 45, 45, 45,
                            45, 48, 53, 45, 6,  45, 38, 45, 38, 45, 16, 45, 38, 16, 45, 45,
                            38, 6,  38, 45, 45, 45, 16, 38, 6,  45, 29, 45, 29, 6,  38, 6};
        vertex_t h_dst[] = {45, 45, 16, 45, 6,  45, 45, 16, 45, 38, 45, 6,  45, 38, 16, 45,
                            45, 45, 45, 53, 29, 16, 45, 8,  8,  16, 45, 38, 45, 6,  45, 45,
                            6,  6,  16, 38, 16, 45, 45, 6,  16, 6,  53, 16, 38, 45, 45, 16};
        weight_t h_wgt[]
            = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

        weight_t h_hubs[]
            = {0, 0,        0,        0, 0, 0, 0.323569, 0, 0.156401, 0, 0,        0,        0, 0,
               0, 0,        0.253312, 0, 0, 0, 0,        0, 0,        0, 0,        0,        0, 0,
               0, 0.110617, 0,        0, 0, 0, 0,        0, 0,        0, 0.365733, 0,        0, 0,
               0, 0,        0,        1, 0, 0, 0.156401, 0, 0,        0, 0,        0.0782005};
        weight_t h_authorities[]
            = {0, 0,         0,        0, 0, 0, 0.321874, 0, 0.123424, 0, 0,        0,       0, 0,
               0, 0,         0.595522, 0, 0, 0, 0,        0, 0,        0, 0,        0,       0, 0,
               0, 0.0292397, 0,        0, 0, 0, 0,        0, 0,        0, 0.314164, 0,       0, 0,
               0, 0,         0,        1, 0, 0, 0,        0, 0,        0, 0,        0.100368};

        double epsilon        = 0.000001;
        size_t max_iterations = 100;

        generic_hits_test<weight_t, vertex_t>(h_src,
                                              h_dst,
                                              h_wgt,
                                              num_vertices,
                                              num_edges,
                                              nullptr,
                                              nullptr,
                                              0,
                                              h_hubs,
                                              h_authorities,
                                              rocgraph_bool_false,
                                              rocgraph_bool_false,
                                              rocgraph_bool_false,
                                              epsilon,
                                              max_iterations);
    }

    void HitsBiggerUnnormalized(const Arguments& arg)
    {
        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 48;
        size_t num_vertices = 54;

        vertex_t h_src[] = {29, 45, 6,  8,  16, 45, 8,  16, 6,  38, 45, 45, 48, 45, 45, 45,
                            45, 48, 53, 45, 6,  45, 38, 45, 38, 45, 16, 45, 38, 16, 45, 45,
                            38, 6,  38, 45, 45, 45, 16, 38, 6,  45, 29, 45, 29, 6,  38, 6};
        vertex_t h_dst[] = {45, 45, 16, 45, 6,  45, 45, 16, 45, 38, 45, 6,  45, 38, 16, 45,
                            45, 45, 45, 53, 29, 16, 45, 8,  8,  16, 45, 38, 45, 6,  45, 45,
                            6,  6,  16, 38, 16, 45, 45, 6,  16, 6,  53, 16, 38, 45, 45, 16};
        weight_t h_wgt[]
            = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

        weight_t h_hubs[]
            = {0, 0,        0,        0, 0, 0, 0.323569, 0, 0.156401, 0, 0,        0,        0, 0,
               0, 0,        0.253312, 0, 0, 0, 0,        0, 0,        0, 0,        0,        0, 0,
               0, 0.110617, 0,        0, 0, 0, 0,        0, 0,        0, 0.365733, 0,        0, 0,
               0, 0,        0,        1, 0, 0, 0.156401, 0, 0,        0, 0,        0.0782005};
        weight_t h_authorities[]
            = {0, 0,         0,        0, 0, 0, 0.321874, 0, 0.123424, 0, 0,        0,       0, 0,
               0, 0,         0.595522, 0, 0, 0, 0,        0, 0,        0, 0,        0,       0, 0,
               0, 0.0292397, 0,        0, 0, 0, 0,        0, 0,        0, 0.314164, 0,       0, 0,
               0, 0,         0,        1, 0, 0, 0,        0, 0,        0, 0,        0.100368};

        double epsilon        = 0.000001;
        size_t max_iterations = 100;

        generic_hits_test<weight_t, vertex_t>(h_src,
                                              h_dst,
                                              h_wgt,
                                              num_vertices,
                                              num_edges,
                                              nullptr,
                                              nullptr,
                                              0,
                                              h_hubs,
                                              h_authorities,
                                              rocgraph_bool_false,
                                              rocgraph_bool_false,
                                              rocgraph_bool_false,
                                              epsilon,
                                              max_iterations);
    }

    void HitsBiggerNormalized(const Arguments& arg)
    {
        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 48;
        size_t num_vertices = 54;

        vertex_t h_src[] = {29, 45, 6,  8,  16, 45, 8,  16, 6,  38, 45, 45, 48, 45, 45, 45,
                            45, 48, 53, 45, 6,  45, 38, 45, 38, 45, 16, 45, 38, 16, 45, 45,
                            38, 6,  38, 45, 45, 45, 16, 38, 6,  45, 29, 45, 29, 6,  38, 6};
        vertex_t h_dst[] = {45, 45, 16, 45, 6,  45, 45, 16, 45, 38, 45, 6,  45, 38, 16, 45,
                            45, 45, 45, 53, 29, 16, 45, 8,  8,  16, 45, 38, 45, 6,  45, 45,
                            6,  6,  16, 38, 16, 45, 45, 6,  16, 6,  53, 16, 38, 45, 45, 16};
        weight_t h_wgt[]
            = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

        weight_t h_hubs[]
            = {0,         0, 0, 0, 0, 0,        0.132381, 0, 0.0639876, 0, 0, 0, 0, 0,         0, 0,
               0.103637,  0, 0, 0, 0, 0,        0,        0, 0,         0, 0, 0, 0, 0.0452563, 0, 0,
               0,         0, 0, 0, 0, 0,        0.149631, 0, 0,         0, 0, 0, 0, 0.409126,  0, 0,
               0.0639876, 0, 0, 0, 0, 0.0319938};

        weight_t h_authorities[]
            = {0,        0,        0, 0, 0, 0, 0.129548, 0, 0.0496755, 0, 0, 0, 0, 0, 0,
               0,        0.239688, 0, 0, 0, 0, 0,        0, 0,         0, 0, 0, 0, 0, 0.0117691,
               0,        0,        0, 0, 0, 0, 0,        0, 0.126445,  0, 0, 0, 0, 0, 0,
               0.402479, 0,        0, 0, 0, 0, 0,        0, 0.0403963};

        double epsilon        = 0.000001;
        size_t max_iterations = 100;

        generic_hits_test<weight_t, vertex_t>(h_src,
                                              h_dst,
                                              h_wgt,
                                              num_vertices,
                                              num_edges,
                                              nullptr,
                                              nullptr,
                                              0,
                                              h_hubs,
                                              h_authorities,
                                              rocgraph_bool_false,
                                              rocgraph_bool_false,
                                              rocgraph_bool_true,
                                              epsilon,
                                              max_iterations);
    }
} // namespace

template <typename T>
void testing_rocgraph_hits_bad_arg(const Arguments& arg)
{
#ifdef TO_FILL

    const rocgraph_handle_t*                        handle{};
    rocgraph_graph_t*                               graph{};
    double                                          epsilon{};
    size_t                                          max_iterations{};
    const rocgraph_type_erased_device_array_view_t* initial_hubs_guess_vertices{};
    const rocgraph_type_erased_device_array_view_t* initial_hubs_guess_values{};
    rocgraph_bool                                   normalize{};
    rocgraph_bool                                   do_expensive_check{};
    rocgraph_hits_result_t**                        result{};
    rocgraph_error_t**                              error{};
    auto                                            ret = rocgraph_hits(handle,
                             graph,
                             epsilon,
                             max_iterations,
                             initial_hubs_guess_vertices,
                             initial_hubs_guess_values,
                             normalize,
                             do_expensive_check,
                             result,
                             error);
    if(ret != rocgraph_status_success)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_internal_error);
    }
    CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
#endif
}

template <typename T>
void testing_rocgraph_hits(const Arguments& arg)
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

#define INSTANTIATE(TYPE)                                                    \
    template void testing_rocgraph_hits_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_rocgraph_hits<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_hits_extra(const Arguments& arg)
{
    testing_dispatch_extra(arg,
                           "Hits",
                           Hits,
                           "HitsWithTranspose",
                           HitsWithTranspose,
                           "HitsWithInitial",
                           HitsWithInitial,
                           "HitsBigger",
                           HitsBigger,
                           "HitsBiggerUnnormalized",
                           HitsBiggerUnnormalized,
                           "HitsBiggerNormalized",
                           HitsBiggerNormalized);
}
