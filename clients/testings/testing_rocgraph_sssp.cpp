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
#include "testing.hpp"
#include "testing_rocgraph_sssp.hpp"

const float EPSILON = 0.001;

namespace
{
    template <typename vertex_t>
    void generic_sssp_test(vertex_t*       h_src,
                           vertex_t*       h_dst,
                           float*          h_wgt,
                           vertex_t        source,
                           float const*    expected_distances,
                           vertex_t const* expected_predecessors,
                           size_t          num_vertices,
                           size_t          num_edges,
                           float           cutoff,
                           rocgraph_bool   store_transposed)
    {

        rocgraph_error_t* ret_error;

        rocgraph_handle_t*       p_handle = nullptr;
        rocgraph_graph_t*        p_graph  = nullptr;
        rocgraph_paths_result_t* p_result = nullptr;

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

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_sssp(p_handle,
                                                   p_graph,
                                                   source,
                                                   cutoff,
                                                   rocgraph_bool_true,
                                                   rocgraph_bool_false,
                                                   &p_result,
                                                   &ret_error),
                                     ret_error);

        rocgraph_type_erased_device_array_view_t* vertices;
        rocgraph_type_erased_device_array_view_t* distances;
        rocgraph_type_erased_device_array_view_t* predecessors;

        vertices     = rocgraph_paths_result_get_vertices(p_result);
        distances    = rocgraph_paths_result_get_distances(p_result);
        predecessors = rocgraph_paths_result_get_predecessors(p_result);

        std::vector<vertex_t> h_vertices(num_vertices);
        std::vector<float>    h_distances(num_vertices);
        std::vector<vertex_t> h_predecessors(num_vertices);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_vertices.data(), vertices, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_distances.data(), distances, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_predecessors.data(), predecessors, &ret_error),
            ret_error);

        ROCGRAPH_CLIENTS_EXPECT_INDIRECT_ARRAY_NEAR_TOLERANCE(num_vertices,
                                                              expected_distances,
                                                              1,
                                                              h_vertices.data(),
                                                              h_distances.data(),
                                                              1,
                                                              (const vertex_t*)nullptr,
                                                              EPSILON);

        ROCGRAPH_CLIENTS_EXPECT_INDIRECT_ARRAY_EQ(num_vertices,
                                                  expected_predecessors,
                                                  1,
                                                  h_vertices.data(),
                                                  h_predecessors.data(),
                                                  1,
                                                  (const vertex_t*)nullptr);

        rocgraph_type_erased_device_array_view_free(vertices);
        rocgraph_type_erased_device_array_view_free(distances);
        rocgraph_type_erased_device_array_view_free(predecessors);
        rocgraph_paths_result_free(p_result);
        rocgraph_sg_graph_free(p_graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    template <typename vertex_t>
    void generic_sssp_test_double(vertex_t*       h_src,
                                  vertex_t*       h_dst,
                                  double*         h_wgt,
                                  vertex_t        source,
                                  double const*   expected_distances,
                                  vertex_t const* expected_predecessors,
                                  size_t          num_vertices,
                                  size_t          num_edges,
                                  double          cutoff,
                                  rocgraph_bool   store_transposed)
    {

        rocgraph_error_t* ret_error;

        rocgraph_handle_t*       p_handle = nullptr;
        rocgraph_graph_t*        p_graph  = nullptr;
        rocgraph_paths_result_t* p_result = nullptr;

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

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_sssp(p_handle,
                                                   p_graph,
                                                   source,
                                                   cutoff,
                                                   rocgraph_bool_true,
                                                   rocgraph_bool_false,
                                                   &p_result,
                                                   &ret_error),
                                     ret_error);

        rocgraph_type_erased_device_array_view_t* vertices;
        rocgraph_type_erased_device_array_view_t* distances;
        rocgraph_type_erased_device_array_view_t* predecessors;

        vertices     = rocgraph_paths_result_get_vertices(p_result);
        distances    = rocgraph_paths_result_get_distances(p_result);
        predecessors = rocgraph_paths_result_get_predecessors(p_result);

        std::vector<vertex_t> h_vertices(num_vertices);
        std::vector<double>   h_distances(num_vertices);
        std::vector<vertex_t> h_predecessors(num_vertices);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_vertices.data(), vertices, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_distances.data(), distances, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_predecessors.data(), predecessors, &ret_error),
            ret_error);

        ROCGRAPH_CLIENTS_EXPECT_INDIRECT_ARRAY_NEAR_TOLERANCE(num_vertices,
                                                              expected_distances,
                                                              1,
                                                              h_vertices.data(),
                                                              h_distances.data(),
                                                              1,
                                                              (const vertex_t*)nullptr,
                                                              EPSILON);

        ROCGRAPH_CLIENTS_EXPECT_INDIRECT_ARRAY_EQ(num_vertices,
                                                  expected_predecessors,
                                                  1,
                                                  h_vertices.data(),
                                                  h_predecessors.data(),
                                                  1,
                                                  (const vertex_t*)nullptr);

        rocgraph_type_erased_device_array_view_free(vertices);
        rocgraph_type_erased_device_array_view_free(distances);
        rocgraph_type_erased_device_array_view_free(predecessors);
        rocgraph_paths_result_free(p_result);
        rocgraph_sg_graph_free(p_graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    void Sssp(const Arguments& arg)
    {
        using vertex_t      = int32_t;
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t src[] = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[] = {1, 3, 4, 0, 1, 3, 5, 5};
        float    wgt[] = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        float    expected_distances[]
            = {0.0f, 0.1f, std::numeric_limits<float>::max(), 2.2f, 1.2f, 4.4f};
        vertex_t expected_predecessors[] = {-1, 0, -1, 1, 1, 4};

        // Bfs wants store_transposed = rocgraph_bool_false
        generic_sssp_test(src,
                          dst,
                          wgt,
                          0,
                          expected_distances,
                          expected_predecessors,
                          num_vertices,
                          num_edges,
                          10,
                          rocgraph_bool_false);
    }

    void SsspWithTranspose(const Arguments& arg)
    {
        using vertex_t      = int32_t;
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t src[] = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[] = {1, 3, 4, 0, 1, 3, 5, 5};
        float    wgt[] = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        float    expected_distances[]
            = {0.0f, 0.1f, std::numeric_limits<float>::max(), 2.2f, 1.2f, 4.4f};
        vertex_t expected_predecessors[] = {-1, 0, -1, 1, 1, 4};

        // Bfs wants store_transposed = rocgraph_bool_false
        //    This call will force rocgraph_sssp to transpose the graph
        generic_sssp_test(src,
                          dst,
                          wgt,
                          0,
                          expected_distances,
                          expected_predecessors,
                          num_vertices,
                          num_edges,
                          10,
                          rocgraph_bool_true);
    }

    void SsspWithTransposeDouble(const Arguments& arg)
    {
        using vertex_t      = int32_t;
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t src[]              = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]              = {1, 3, 4, 0, 1, 3, 5, 5};
        double   wgt[]              = {0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2};
        double expected_distances[] = {0.0, 0.1, std::numeric_limits<double>::max(), 2.2, 1.2, 4.4};
        vertex_t expected_predecessors[] = {-1, 0, -1, 1, 1, 4};

        // Bfs wants store_transposed = rocgraph_bool_false
        //    This call will force rocgraph_sssp to transpose the graph
        generic_sssp_test_double(src,
                                 dst,
                                 wgt,
                                 0,
                                 expected_distances,
                                 expected_predecessors,
                                 num_vertices,
                                 num_edges,
                                 10,
                                 rocgraph_bool_true);
    }

} // namespace

template <typename T>
void testing_rocgraph_sssp_bad_arg(const Arguments& arg)
{
#ifdef TO_FILL
    const rocgraph_handle_t*  handle{};
    rocgraph_graph_t*         graph{};
    size_t                    source{};
    double                    cutoff{};
    rocgraph_bool             compute_predecessors{};
    rocgraph_bool             do_expensive_check{};
    rocgraph_paths_result_t** result{};
    rocgraph_error_t**        error{};
    auto                      ret = rocgraph_sssp(
        handle, graph, source, cutoff, compute_predecessors, do_expensive_check, result, error);
    if(ret != rocgraph_status_success)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_internal_error);
    }
    CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
#endif
}

template <typename T>
void testing_rocgraph_sssp(const Arguments& arg)
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
    template void testing_rocgraph_sssp_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_rocgraph_sssp<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_sssp_extra(const Arguments& arg)
{
    testing_dispatch_extra(arg,
                           "Sssp",
                           Sssp,
                           "SsspWithTranspose",
                           SsspWithTranspose,
                           "SsspWithTransposeDouble",
                           SsspWithTransposeDouble);
}
