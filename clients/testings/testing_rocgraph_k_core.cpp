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

namespace
{
    template <typename weight_t, typename vertex_t>
    static void generic_k_core_test(vertex_t*     h_src,
                                    vertex_t*     h_dst,
                                    weight_t*     h_wgt,
                                    vertex_t*     h_result_src,
                                    vertex_t*     h_result_dst,
                                    weight_t*     h_result_wgt,
                                    size_t        num_vertices,
                                    size_t        num_edges,
                                    size_t        num_result_edges,
                                    size_t        k,
                                    rocgraph_bool store_transposed)
    {
        rocgraph_error_t* ret_error;

        rocgraph_data_type_id vertex_tid    = rocgraph_data_type_id_int32;
        rocgraph_data_type_id edge_tid      = rocgraph_data_type_id_int32;
        rocgraph_data_type_id weight_tid    = rocgraph_data_type_id_float32;
        rocgraph_data_type_id edge_id_tid   = rocgraph_data_type_id_int32;
        rocgraph_data_type_id edge_type_tid = rocgraph_data_type_id_int32;

        rocgraph_handle_t*        p_handle      = nullptr;
        rocgraph_graph_t*         graph         = nullptr;
        rocgraph_core_result_t*   core_result   = nullptr;
        rocgraph_k_core_result_t* k_core_result = nullptr;

        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        rocgraph_clients_create_sg_test_graph(p_handle,
                                              vertex_tid,
                                              edge_tid,
                                              h_src,
                                              h_dst,
                                              weight_tid,
                                              h_wgt,
                                              edge_type_tid,
                                              nullptr,
                                              edge_id_tid,
                                              nullptr,
                                              num_edges,
                                              store_transposed,
                                              rocgraph_bool_false,
                                              rocgraph_bool_true,
                                              rocgraph_bool_false,
                                              &graph,
                                              &ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_core_number(p_handle,
                                                          graph,
                                                          rocgraph_k_core_degree_type_in,
                                                          rocgraph_bool_false,
                                                          &core_result,
                                                          &ret_error),
                                     ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_k_core(p_handle,
                                                     graph,
                                                     k,
                                                     rocgraph_k_core_degree_type_in,
                                                     core_result,
                                                     rocgraph_bool_false,
                                                     &k_core_result,
                                                     &ret_error),
                                     ret_error);

        rocgraph_type_erased_device_array_view_t* src_vertices;
        rocgraph_type_erased_device_array_view_t* dst_vertices;
        rocgraph_type_erased_device_array_view_t* weights;

        src_vertices = rocgraph_k_core_result_get_src_vertices(k_core_result);
        dst_vertices = rocgraph_k_core_result_get_dst_vertices(k_core_result);
        weights      = rocgraph_k_core_result_get_weights(k_core_result);

        size_t number_of_result_edges = rocgraph_type_erased_device_array_view_size(src_vertices);

        std::vector<vertex_t> h_src_vertices(number_of_result_edges);
        std::vector<vertex_t> h_dst_vertices(number_of_result_edges);
        std::vector<weight_t> h_weights(number_of_result_edges);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_src_vertices.data(), src_vertices, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_dst_vertices.data(), dst_vertices, &ret_error),
            ret_error);

        if(weights != nullptr)
        {
            CHECK_ROCGRAPH_SUCCESS_ERROR(
                rocgraph_type_erased_device_array_view_copy_to_host(
                    p_handle, (rocgraph_byte_t*)h_weights.data(), weights, &ret_error),
                ret_error);
        }

        ROCGRAPH_CLIENTS_EXPECT_EQ(number_of_result_edges, num_result_edges);

        std::vector<weight_t> M(num_vertices * num_vertices, weight_t(0));

        for(int i = 0; i < num_result_edges; ++i)
            M[h_result_src[i] + num_vertices * h_result_dst[i]]
                = (h_result_wgt != nullptr) ? h_result_wgt[i] : 1.0;

        for(int i = 0; i < number_of_result_edges; ++i)
        {
            ROCGRAPH_CLIENTS_EXPECT_EQ(M[h_src_vertices[i] + num_vertices * h_dst_vertices[i]],
                                       (h_result_wgt != nullptr) ? h_weights[i] : weight_t(1.0));
        }

        rocgraph_k_core_result_free(k_core_result);
        rocgraph_core_result_free(core_result);
        rocgraph_sg_graph_free(graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }
    void KCore(const Arguments& arg)
    {

        using vertex_t          = int32_t;
        using weight_t          = float;
        size_t num_edges        = 22;
        size_t num_vertices     = 7;
        size_t num_result_edges = 12;
        size_t k                = 3;

        vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5, 3, 1, 4, 5, 5, 6};
        vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 3, 1, 6, 5};
        weight_t h_wgt[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        vertex_t h_result_src[] = {1, 1, 3, 4, 3, 4, 3, 4, 5, 5, 1, 5};
        vertex_t h_result_dst[] = {3, 4, 5, 5, 1, 3, 4, 1, 3, 4, 5, 1};
        weight_t h_result_wgt[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

        generic_k_core_test<weight_t, vertex_t>(h_src,
                                                h_dst,
                                                h_wgt,
                                                h_result_src,
                                                h_result_dst,
                                                h_result_wgt,
                                                num_vertices,
                                                num_edges,
                                                num_result_edges,
                                                k,
                                                rocgraph_bool_false);
    }

    void KCoreNoWeights(const Arguments& arg)
    {

        using vertex_t          = int32_t;
        using weight_t          = float;
        size_t num_edges        = 22;
        size_t num_vertices     = 7;
        size_t num_result_edges = 12;
        size_t k                = 3;

        vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5, 3, 1, 4, 5, 5, 6};
        vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 3, 1, 6, 5};
        vertex_t h_result_src[] = {1, 1, 3, 4, 3, 4, 3, 4, 5, 5, 1, 5};
        vertex_t h_result_dst[] = {3, 4, 5, 5, 1, 3, 4, 1, 3, 4, 5, 1};

        generic_k_core_test<weight_t, vertex_t>(h_src,
                                                h_dst,
                                                nullptr,
                                                h_result_src,
                                                h_result_dst,
                                                nullptr,
                                                num_vertices,
                                                num_edges,
                                                num_result_edges,
                                                k,
                                                rocgraph_bool_false);
    }

} // namespace

#include "rocgraph/rocgraph.h"
#include "testing.hpp"
#include "testing_rocgraph_k_core.hpp"

template <typename T>
void testing_rocgraph_k_core_bad_arg(const Arguments& arg)
{
#ifdef TO_FILL
    const rocgraph_handle_t*      handle{};
    rocgraph_graph_t*             graph{};
    size_t                        k{};
    rocgraph_k_core_degree_type   degree_type{};
    const rocgraph_core_result_t* core_result{};
    rocgraph_bool                 do_expensive_check{};
    rocgraph_k_core_result_t**    result{};
    rocgraph_error_t**            error{};
    auto                          ret = rocgraph_k_core(
        handle, graph, k, degree_type, core_result, do_expensive_check, result, error);
    if(ret != rocgraph_status_success)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_internal_error);
    }
    CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
#endif
}

template <typename T>
void testing_rocgraph_k_core(const Arguments& arg)
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

#define INSTANTIATE(TYPE)                                                      \
    template void testing_rocgraph_k_core_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_rocgraph_k_core<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_k_core_extra(const Arguments& arg)
{
    testing_dispatch_extra(arg, "KCore", KCore, "KCoreNoWeights", KCoreNoWeights);
}
