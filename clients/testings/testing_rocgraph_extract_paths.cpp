// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/*
 * Copyright (C) 2021-2024, NVIDIA CORPORATION.
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

#include "testing_rocgraph_extract_paths.hpp"
#include "rocgraph/rocgraph.h"
#include "rocgraph_clients_create_test_graph.hpp"
#include "rocgraph_test.hpp"
#include "testing.hpp"

namespace
{

    template <typename weight_t, typename vertex_t>
    static void generic_bfs_test_with_extract_paths(vertex_t*       h_src,
                                                    vertex_t*       h_dst,
                                                    weight_t*       h_wgt,
                                                    vertex_t*       h_seeds,
                                                    vertex_t*       h_destinations,
                                                    vertex_t        expected_max_path_length,
                                                    vertex_t const* expected_paths,
                                                    size_t          num_vertices,
                                                    size_t          num_edges,
                                                    size_t          num_seeds,
                                                    size_t          num_destinations,
                                                    size_t          depth_limit,
                                                    rocgraph_bool   store_transposed)
    {
        rocgraph_error_t* ret_error;

        rocgraph_handle_t*                        p_handle               = nullptr;
        rocgraph_graph_t*                         p_graph                = nullptr;
        rocgraph_paths_result_t*                  p_paths_result         = nullptr;
        rocgraph_extract_paths_result_t*          p_extract_paths_result = nullptr;
        rocgraph_type_erased_device_array_t*      p_sources              = nullptr;
        rocgraph_type_erased_device_array_t*      p_destinations         = nullptr;
        rocgraph_type_erased_device_array_view_t* p_sources_view         = nullptr;
        rocgraph_type_erased_device_array_view_t* p_destinations_view    = nullptr;

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

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_create(
                p_handle, num_seeds, rocgraph_data_type_id_int32, &p_sources, &ret_error),
            ret_error);

        p_sources_view = rocgraph_type_erased_device_array_view(p_sources);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_from_host(
                p_handle, p_sources_view, (rocgraph_byte_t*)h_seeds, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_create(p_handle,
                                                     num_destinations,
                                                     rocgraph_data_type_id_int32,
                                                     &p_destinations,
                                                     &ret_error),
            ret_error);

        p_destinations_view = rocgraph_type_erased_device_array_view(p_destinations);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_from_host(
                p_handle, p_destinations_view, (rocgraph_byte_t*)h_destinations, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_bfs(p_handle,
                                                  p_graph,
                                                  p_sources_view,
                                                  rocgraph_bool_false,
                                                  depth_limit,
                                                  rocgraph_bool_true,
                                                  rocgraph_bool_false,
                                                  &p_paths_result,
                                                  &ret_error),
                                     ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_extract_paths(p_handle,
                                                            p_graph,
                                                            p_sources_view,
                                                            p_paths_result,
                                                            p_destinations_view,
                                                            &p_extract_paths_result,
                                                            &ret_error),
                                     ret_error);

        size_t max_path_length
            = rocgraph_extract_paths_result_get_max_path_length(p_extract_paths_result);

        ROCGRAPH_CLIENTS_EXPECT_EQ(max_path_length, size_t(expected_max_path_length));

        rocgraph_type_erased_device_array_view_t* paths_view
            = rocgraph_extract_paths_result_get_paths(p_extract_paths_result);

        size_t paths_size = rocgraph_type_erased_device_array_view_size(paths_view);

        std::vector<vertex_t> h_paths(paths_size);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_paths.data(), paths_view, &ret_error),
            ret_error);

        ROCGRAPH_CLIENTS_EXPECT_SEGMENT_EQ(paths_size, expected_paths, h_paths.data());

        rocgraph_type_erased_device_array_view_free(paths_view);
        rocgraph_type_erased_device_array_view_free(p_sources_view);
        rocgraph_type_erased_device_array_view_free(p_destinations_view);
        rocgraph_type_erased_device_array_free(p_sources);
        rocgraph_type_erased_device_array_free(p_destinations);
        rocgraph_extract_paths_result_free(p_extract_paths_result);
        rocgraph_paths_result_free(p_paths_result);
        rocgraph_sg_graph_free(p_graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    void BfsWithExtractPaths(const Arguments& arg)
    {

        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t src[]                    = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]                    = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t wgt[]                    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t seeds[]                  = {0};
        vertex_t destinations[]           = {5};
        vertex_t expected_max_path_length = 4;
        vertex_t expected_paths[]         = {0, 1, 3, 5};

        // Bfs wants store_transposed = rocgraph_bool_false
        generic_bfs_test_with_extract_paths(src,
                                            dst,
                                            wgt,
                                            seeds,
                                            destinations,
                                            expected_max_path_length,
                                            expected_paths,
                                            num_vertices,
                                            num_edges,
                                            1,
                                            1,
                                            10,
                                            rocgraph_bool_false);
    }

    void BfsWithExtractPathsWithTranspose(const Arguments& arg)
    {

        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t src[]                    = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]                    = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t wgt[]                    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t seeds[]                  = {0};
        vertex_t destinations[]           = {5};
        vertex_t expected_max_path_length = 4;
        vertex_t expected_paths[]         = {0, 1, 3, 5};

        // Bfs wants store_transposed = rocgraph_bool_false
        //    This call will force rocgraph_bfs to transpose the graph
        generic_bfs_test_with_extract_paths(src,
                                            dst,
                                            wgt,
                                            seeds,
                                            destinations,
                                            expected_max_path_length,
                                            expected_paths,
                                            num_vertices,
                                            num_edges,
                                            1,
                                            1,
                                            10,
                                            rocgraph_bool_true);
    }

} // namespace

template <typename T>
void testing_rocgraph_extract_paths_bad_arg(const Arguments& arg)
{
#ifdef TO_FILL

    const rocgraph_handle_t*                        handle{};
    rocgraph_graph_t*                               graph{};
    const rocgraph_type_erased_device_array_view_t* sources{};
    const rocgraph_paths_result_t*                  paths_result{};
    const rocgraph_type_erased_device_array_view_t* destinations{};
    rocgraph_extract_paths_result_t**               result{};
    rocgraph_error_t**                              error{};
    auto                                            ret
        = rocgraph_extract_paths(handle, graph, sources, paths_result, destinations, result, error);
    if(ret != rocgraph_status_success)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_internal_error);
    }
    CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
#endif
}

template <typename T>
void testing_rocgraph_extract_paths(const Arguments& arg)
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

#define INSTANTIATE(TYPE)                                                             \
    template void testing_rocgraph_extract_paths_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_rocgraph_extract_paths<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_extract_paths_extra(const Arguments& arg)
{
    testing_dispatch_extra(arg,
                           "BfsWithExtractPaths",
                           BfsWithExtractPaths,
                           "BfsWithExtractPathsWithTranspose",
                           BfsWithExtractPathsWithTranspose);
}
