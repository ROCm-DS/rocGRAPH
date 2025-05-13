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

#include "testing_rocgraph_two_hop_neighbors.hpp"
#include "rocgraph/internal/rocgraph_graph_functions.h"
#include "rocgraph/rocgraph.h"
#include "rocgraph_clients_create_test_graph.hpp"
#include "rocgraph_test.hpp"
#include "testing.hpp"

namespace
{
    template <typename weight_t, typename vertex_t>
    void generic_two_hop_nbr_test(vertex_t*     h_src,
                                  vertex_t*     h_dst,
                                  weight_t*     h_wgt,
                                  vertex_t*     h_sources,
                                  vertex_t*     h_result_v1,
                                  vertex_t*     h_result_v2,
                                  size_t        num_vertices,
                                  size_t        num_edges,
                                  size_t        num_sources,
                                  size_t        num_result_pairs,
                                  rocgraph_bool store_transposed)
    {
        rocgraph_error_t* ret_error;

        rocgraph_handle_t*                        p_handle            = nullptr;
        rocgraph_graph_t*                         graph               = nullptr;
        rocgraph_type_erased_device_array_t*      start_vertices      = nullptr;
        rocgraph_type_erased_device_array_view_t* start_vertices_view = nullptr;
        rocgraph_vertex_pairs_t*                  result              = nullptr;

        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        rocgraph_clients_create_test_graph(p_handle,
                                           h_src,
                                           h_dst,
                                           h_wgt,
                                           num_edges,
                                           store_transposed,
                                           rocgraph_bool_false,
                                           rocgraph_bool_true,
                                           &graph,
                                           &ret_error);

        if(num_sources > 0)
        {
            CHECK_ROCGRAPH_SUCCESS_ERROR(
                rocgraph_type_erased_device_array_create(p_handle,
                                                         num_sources,
                                                         rocgraph_data_type_id_int32,
                                                         &start_vertices,
                                                         &ret_error),
                ret_error);
            start_vertices_view = rocgraph_type_erased_device_array_view(start_vertices);

            CHECK_ROCGRAPH_SUCCESS_ERROR(
                rocgraph_type_erased_device_array_view_copy_from_host(
                    p_handle, start_vertices_view, (rocgraph_byte_t*)h_sources, &ret_error),
                ret_error);
        }

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_two_hop_neighbors(
                p_handle, graph, start_vertices_view, rocgraph_bool_false, &result, &ret_error),
            ret_error);

        rocgraph_type_erased_device_array_view_t const* v1;
        rocgraph_type_erased_device_array_view_t const* v2;

        v1 = rocgraph_vertex_pairs_get_first(result);
        v2 = rocgraph_vertex_pairs_get_second(result);

        size_t number_of_pairs = rocgraph_type_erased_device_array_view_size(v1);

        std::vector<vertex_t> h_v1(number_of_pairs);
        std::vector<vertex_t> h_v2(number_of_pairs);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_to_host(
                                         p_handle, (rocgraph_byte_t*)h_v1.data(), v1, &ret_error),
                                     ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_to_host(
                                         p_handle, (rocgraph_byte_t*)h_v2.data(), v2, &ret_error),
                                     ret_error);

        ROCGRAPH_CLIENTS_EXPECT_EQ(number_of_pairs, num_result_pairs);

        std::vector<bool> M(num_vertices * num_vertices, false);
        for(int i = 0; i < num_result_pairs; ++i)
        {
            M[h_result_v1[i] + num_vertices * h_result_v2[i]] = true;
        }

        for(int i = 0; i < number_of_pairs; ++i)
        {
            const bool val = M[h_v1[i] + num_vertices * h_v2[i]];
            ROCGRAPH_CLIENTS_EXPECT_TRUE(val);
        }

        rocgraph_vertex_pairs_free(result);
        rocgraph_type_erased_device_array_view_free(start_vertices_view);
        rocgraph_type_erased_device_array_free(start_vertices);
        rocgraph_sg_graph_free(graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }
    void TwoHopNbrAll(const Arguments& arg)
    {

        using vertex_t = int32_t;
        using weight_t = float;

        size_t num_edges        = 22;
        size_t num_vertices     = 7;
        size_t num_sources      = 0;
        size_t num_result_pairs = 43;

        vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5, 3, 1, 4, 5, 5, 6};
        vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 3, 1, 6, 5};
        weight_t h_wgt[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

        vertex_t h_result_v1[] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3,
                                  3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6};
        vertex_t h_result_v2[] = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 0, 1, 2,
                                  3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 1, 3, 4, 6};

        generic_two_hop_nbr_test<weight_t, vertex_t>(h_src,
                                                     h_dst,
                                                     h_wgt,
                                                     nullptr,
                                                     h_result_v1,
                                                     h_result_v2,
                                                     num_vertices,
                                                     num_edges,
                                                     num_sources,
                                                     num_result_pairs,
                                                     rocgraph_bool_false);
    }

    void TwoHopNbrOne(const Arguments& arg)
    {

        using vertex_t = int32_t;
        using weight_t = float;

        size_t num_edges        = 22;
        size_t num_vertices     = 7;
        size_t num_sources      = 1;
        size_t num_result_pairs = 6;

        vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5, 3, 1, 4, 5, 5, 6};
        vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 3, 1, 6, 5};
        weight_t h_wgt[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

        vertex_t h_sources[] = {0};

        vertex_t h_result_v1[] = {0, 0, 0, 0, 0, 0};
        vertex_t h_result_v2[] = {0, 1, 2, 3, 4, 5};

        generic_two_hop_nbr_test<weight_t, vertex_t>(h_src,
                                                     h_dst,
                                                     h_wgt,
                                                     h_sources,
                                                     h_result_v1,
                                                     h_result_v2,
                                                     num_vertices,
                                                     num_edges,
                                                     num_sources,
                                                     num_result_pairs,
                                                     rocgraph_bool_false);
    }

} // namespace
template <typename T>
void testing_rocgraph_two_hop_neighbors_bad_arg(const Arguments& arg)
{
#ifdef TO_FILL
    const rocgraph_handle_t*                        handle{};
    rocgraph_graph_t*                               graph{};
    const rocgraph_type_erased_device_array_view_t* start_vertices{};
    rocgraph_bool                                   do_expensive_check{};
    rocgraph_vertex_pairs_t**                       result{};
    rocgraph_error_t**                              error{};
    auto                                            ret = rocgraph_two_hop_neighbors(
        handle, graph, start_vertices, do_expensive_check, result, error);
    if(ret != rocgraph_status_success)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_internal_error);
    }
    CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
#endif
}

template <typename T>
void testing_rocgraph_two_hop_neighbors(const Arguments& arg)
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

#define INSTANTIATE(TYPE)                                                                 \
    template void testing_rocgraph_two_hop_neighbors_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_rocgraph_two_hop_neighbors<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_two_hop_neighbors_extra(const Arguments& arg)
{
    testing_dispatch_extra(arg, "TwoHopNbrOne", TwoHopNbrOne, "TwoHopNbrAll", TwoHopNbrAll);
}
