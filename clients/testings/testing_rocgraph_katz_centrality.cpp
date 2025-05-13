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

#include "testing_rocgraph_katz_centrality.hpp"
#include "rocgraph/rocgraph.h"
#include "rocgraph_clients_create_test_graph.hpp"
#include "rocgraph_test.hpp"
#include "testing.hpp"

namespace
{
    template <typename weight_t, typename vertex_t>
    static void generic_katz_test(vertex_t*     h_src,
                                  vertex_t*     h_dst,
                                  weight_t*     h_wgt,
                                  weight_t*     h_result,
                                  size_t        num_vertices,
                                  size_t        num_edges,
                                  rocgraph_bool store_transposed,
                                  double        alpha,
                                  double        beta,
                                  double        epsilon,
                                  size_t        max_iterations)
    {
        rocgraph_error_t* ret_error;

        rocgraph_handle_t*                        p_handle   = nullptr;
        rocgraph_graph_t*                         p_graph    = nullptr;
        rocgraph_centrality_result_t*             p_result   = nullptr;
        rocgraph_type_erased_device_array_view_t* betas_view = nullptr;

        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        rocgraph_clients_create_test_graph(p_handle,
                                           h_src,
                                           h_dst,
                                           h_wgt,
                                           num_edges,
                                           store_transposed,
                                           rocgraph_bool_false,
                                           rocgraph_bool_false,
                                           &p_graph,
                                           &ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_katz_centrality(p_handle,
                                                              p_graph,
                                                              betas_view,
                                                              alpha,
                                                              beta,
                                                              epsilon,
                                                              max_iterations,
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
                                                              0.001);

        rocgraph_centrality_result_free(p_result);
        rocgraph_sg_graph_free(p_graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    void Katz(const Arguments& arg)
    {

        using vertex_t = int32_t;
        using weight_t = float;

        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        weight_t h_result[] = {0.410614, 0.403211, 0.390689, 0.415175, 0.395125, 0.433226};

        double alpha          = 0.01;
        double beta           = 1.0;
        double epsilon        = 0.000001;
        size_t max_iterations = 1000;

        // Katz wants store_transposed = rocgraph_bool_true
        generic_katz_test<weight_t, vertex_t>(h_src,
                                              h_dst,
                                              h_wgt,
                                              h_result,
                                              num_vertices,
                                              num_edges,
                                              rocgraph_bool_true,
                                              alpha,
                                              beta,
                                              epsilon,
                                              max_iterations);
    }

} // namespace

template <typename T>
void testing_rocgraph_katz_centrality_bad_arg(const Arguments& arg)
{
#ifdef TO_FILL
    const rocgraph_handle_t*                        handle{};
    rocgraph_graph_t*                               graph{};
    const rocgraph_type_erased_device_array_view_t* betas{};
    double                                          alpha{};
    double                                          beta{};
    double                                          epsilon{};
    size_t                                          max_iterations{};
    rocgraph_bool                                   do_expensive_check{};
    rocgraph_centrality_result_t**                  result{};
    rocgraph_error_t**                              error{};
    auto                                            ret = rocgraph_katz_centrality(handle,
                                        graph,
                                        betas,
                                        alpha,
                                        beta,
                                        epsilon,
                                        max_iterations,
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
void testing_rocgraph_katz_centrality(const Arguments& arg)
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

#define INSTANTIATE(TYPE)                                                               \
    template void testing_rocgraph_katz_centrality_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_rocgraph_katz_centrality<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_katz_centrality_extra(const Arguments& arg)
{
    testing_dispatch_extra(arg, "Katz", Katz);
}
