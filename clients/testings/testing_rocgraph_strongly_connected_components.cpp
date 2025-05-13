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

#include "testing_rocgraph_strongly_connected_components.hpp"
#include "rocgraph/rocgraph.h"
#include "rocgraph_clients_create_test_graph.hpp"
#include "rocgraph_test.hpp"
#include "testing.hpp"

namespace
{

    template <typename weight_t, typename vertex_t>
    void generic_scc_test(vertex_t*     h_src,
                          vertex_t*     h_dst,
                          weight_t*     h_wgt,
                          vertex_t*     h_result,
                          size_t        num_vertices,
                          size_t        num_edges,
                          rocgraph_bool store_transposed)
    {

        rocgraph_error_t* ret_error = nullptr;

        rocgraph_handle_t*          p_handle = nullptr;
        rocgraph_graph_t*           p_graph  = nullptr;
        rocgraph_labeling_result_t* p_result = nullptr;

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
            rocgraph_strongly_connected_components(
                p_handle, p_graph, rocgraph_bool_false, &p_result, &ret_error),
            ret_error);

        rocgraph_type_erased_device_array_view_t* vertices;
        rocgraph_type_erased_device_array_view_t* components;

        vertices   = rocgraph_labeling_result_get_vertices(p_result);
        components = rocgraph_labeling_result_get_labels(p_result);

        std::vector<vertex_t> h_vertices(num_vertices);
        std::vector<vertex_t> h_components(num_vertices);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_vertices.data(), vertices, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_components.data(), components, &ret_error),
            ret_error);

        std::vector<vertex_t> component_check(num_vertices, num_vertices);
        for(vertex_t i = 0; i < num_vertices; ++i)
        {
            if(component_check[h_result[i]] == num_vertices)
                component_check[h_result[i]] = h_components[i];
        }

        ROCGRAPH_CLIENTS_EXPECT_INDIRECT_ARRAY_EQ(num_vertices,
                                                  h_components.data(),
                                                  1,
                                                  (const vertex_t*)nullptr,
                                                  component_check.data(),
                                                  1,
                                                  h_result);

        rocgraph_type_erased_device_array_view_free(components);
        rocgraph_type_erased_device_array_view_free(vertices);
        rocgraph_labeling_result_free(p_result);

        rocgraph_sg_graph_free(p_graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    void StronglyConnectedComponents(const Arguments& arg)
    {

        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 19;
        size_t num_vertices = 12;

        vertex_t h_src[] = {0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 6, 7, 7, 8, 8, 8, 9, 10};
        vertex_t h_dst[] = {1, 2, 3, 4, 0, 1, 3, 4, 5, 3, 5, 7, 9, 10, 6, 7, 9, 11, 11};
        weight_t h_wgt[] = {1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0};

        vertex_t h_result[] = {0, 0, 0, 3, 3, 5, 6, 7, 8, 9, 10, 11};

        // SCC wants store_transposed = rocgraph_bool_false
        generic_scc_test(
            h_src, h_dst, h_wgt, h_result, num_vertices, num_edges, rocgraph_bool_false);
    }

} // namespace

template <typename T>
void testing_rocgraph_strongly_connected_components_bad_arg(const Arguments& arg)
{
#ifdef TOFILL

    const rocgraph_handle_t*     handle{};
    rocgraph_graph_t*            graph{};
    rocgraph_bool                do_expensive_check{};
    rocgraph_labeling_result_t** result{};
    rocgraph_error_t**           error{};
    auto                         ret
        = rocgraph_strongly_connected_components(handle, graph, do_expensive_check, result, error);
    if(ret != rocgraph_status_success)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_internal_error);
    }
    CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
#endif
}

template <typename T>
void testing_rocgraph_strongly_connected_components(const Arguments& arg)
{
#ifdef TOFILL

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

#define INSTANTIATE(TYPE)                                                       \
    template void testing_rocgraph_strongly_connected_components_bad_arg<TYPE>( \
        const Arguments& arg);                                                  \
    template void testing_rocgraph_strongly_connected_components<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_strongly_connected_components_extra(const Arguments& arg)
{
    testing_dispatch_extra(arg, "StronglyConnectedComponents", StronglyConnectedComponents);
}
