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
#include "rocgraph_clients_expect_array_eq.hpp"
#include "rocgraph_clients_expect_eq.hpp"
#include "rocgraph_test.hpp"

#include "rocgraph/rocgraph.h"
#include "testing.hpp"
#include "testing_rocgraph_extract_ego.hpp"

namespace
{
    template <typename weight_t, typename vertex_t>
    static void generic_egonet_test(vertex_t*     h_src,
                                    vertex_t*     h_dst,
                                    weight_t*     h_wgt,
                                    vertex_t*     h_seeds,
                                    vertex_t*     h_expected_src,
                                    vertex_t*     h_expected_dst,
                                    size_t*       h_expected_offsets,
                                    size_t        num_vertices,
                                    size_t        num_edges,
                                    size_t        num_seeds,
                                    size_t        radius,
                                    rocgraph_bool store_transposed)
    {
        rocgraph_error_t* ret_error;

        rocgraph_data_type_id vertex_tid    = rocgraph_data_type_id_int32;
        rocgraph_data_type_id edge_tid      = rocgraph_data_type_id_int32;
        rocgraph_data_type_id weight_tid    = rocgraph_data_type_id_float32;
        rocgraph_data_type_id edge_id_tid   = rocgraph_data_type_id_int32;
        rocgraph_data_type_id edge_type_tid = rocgraph_data_type_id_int32;

        rocgraph_handle_t*                        p_handle   = nullptr;
        rocgraph_graph_t*                         graph      = nullptr;
        rocgraph_type_erased_device_array_t*      seeds      = nullptr;
        rocgraph_type_erased_device_array_view_t* seeds_view = nullptr;
        rocgraph_induced_subgraph_result_t*       result     = nullptr;

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
                                              rocgraph_bool_false,
                                              rocgraph_bool_false,
                                              &graph,
                                              &ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_create(
                p_handle, num_seeds, rocgraph_data_type_id_int32, &seeds, &ret_error),
            ret_error);
        seeds_view = rocgraph_type_erased_device_array_view(seeds);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_from_host(
                p_handle, seeds_view, (rocgraph_byte_t*)h_seeds, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_extract_ego(
                p_handle, graph, seeds_view, radius, rocgraph_bool_false, &result, &ret_error),
            ret_error);

        rocgraph_type_erased_device_array_view_t* src;
        rocgraph_type_erased_device_array_view_t* dst;
        // rocgraph_type_erased_device_array_view_t* wgt;
        rocgraph_type_erased_device_array_view_t* offsets;

        src = rocgraph_induced_subgraph_get_sources(result);
        dst = rocgraph_induced_subgraph_get_destinations(result);
        // wgt     = rocgraph_induced_subgraph_get_edge_weights(result);
        offsets = rocgraph_induced_subgraph_get_subgraph_offsets(result);

        size_t num_result_edges   = rocgraph_type_erased_device_array_view_size(src);
        size_t num_result_offsets = rocgraph_type_erased_device_array_view_size(offsets);

        std::vector<vertex_t> h_result_src(num_result_edges);
        std::vector<vertex_t> h_result_dst(num_result_edges);
        // weight_t h_result_wgt[num_result_edges];
        std::vector<size_t> h_result_offsets(num_result_offsets);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_src.data(), src, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_dst.data(), dst, &ret_error),
            ret_error);

        // CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_to_host(
        //     p_handle, (rocgraph_byte_t*)h_result_wgt, wgt, &ret_error),
        // ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_offsets.data(), offsets, &ret_error),
            ret_error);

        ROCGRAPH_CLIENTS_EXPECT_EQ((num_seeds + 1), num_result_offsets);

        ROCGRAPH_CLIENTS_EXPECT_SEGMENT_EQ(
            num_result_offsets, (const size_t*)h_result_offsets.data(), h_expected_offsets);

        std::vector<weight_t> M(num_vertices * num_vertices, weight_t(0));
        for(int i = 0; i < num_seeds; ++i)
        {
            for(size_t e = h_expected_offsets[i]; e < h_expected_offsets[i + 1]; ++e)
            {
                M[h_expected_src[e] + num_vertices * h_expected_dst[e]] = 1.0;
            }

            for(size_t e = h_result_offsets[i]; e < h_result_offsets[i + 1]; ++e)
            {
                ROCGRAPH_CLIENTS_EXPECT_GT(M[h_result_src[e] + num_vertices * h_result_dst[e]],
                                           0.0f);
            }
        }

        rocgraph_type_erased_device_array_view_free(src);
        rocgraph_type_erased_device_array_view_free(dst);
        // rocgraph_type_erased_device_array_view_free(wgt);
        rocgraph_type_erased_device_array_view_free(offsets);
        rocgraph_induced_subgraph_result_free(result);

        rocgraph_sg_graph_free(graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    static void Egonet(const Arguments& arg)
    {

        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 9;
        size_t num_vertices = 6;
        size_t radius       = 2;
        size_t num_seeds    = 2;

        vertex_t h_src[]   = {0, 1, 1, 2, 2, 2, 3, 3, 4};
        vertex_t h_dst[]   = {1, 3, 4, 0, 1, 3, 4, 5, 5};
        weight_t h_wgt[]   = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 6.1f};
        vertex_t h_seeds[] = {0, 1};

        vertex_t h_result_src[]     = {0, 1, 1, 3, 1, 1, 3, 3, 4};
        vertex_t h_result_dst[]     = {1, 3, 4, 4, 3, 4, 4, 5, 5};
        size_t   h_result_offsets[] = {0, 4, 9};

        // Egonet wants store_transposed = rocgraph_bool_false
        generic_egonet_test<weight_t, vertex_t>(h_src,
                                                h_dst,
                                                h_wgt,
                                                h_seeds,
                                                h_result_src,
                                                h_result_dst,
                                                h_result_offsets,
                                                num_vertices,
                                                num_edges,
                                                num_seeds,
                                                radius,
                                                rocgraph_bool_false);
    }

    static void EgonetNoWeights(const Arguments& arg)
    {

        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 9;
        size_t num_vertices = 6;
        size_t radius       = 2;
        size_t num_seeds    = 2;

        vertex_t h_src[]   = {0, 1, 1, 2, 2, 2, 3, 3, 4};
        vertex_t h_dst[]   = {1, 3, 4, 0, 1, 3, 4, 5, 5};
        vertex_t h_seeds[] = {0, 1};

        vertex_t h_result_src[]     = {0, 1, 1, 3, 1, 1, 3, 3, 4};
        vertex_t h_result_dst[]     = {1, 3, 4, 4, 3, 4, 4, 5, 5};
        size_t   h_result_offsets[] = {0, 4, 9};

        // Egonet wants store_transposed = rocgraph_bool_false
        generic_egonet_test<weight_t, vertex_t>(h_src,
                                                h_dst,
                                                nullptr,
                                                h_seeds,
                                                h_result_src,
                                                h_result_dst,
                                                h_result_offsets,
                                                num_vertices,
                                                num_edges,
                                                num_seeds,
                                                radius,
                                                rocgraph_bool_false);
    }

} // namespace

template <typename T>
void testing_rocgraph_extract_ego_bad_arg(const Arguments& arg)
{
#ifdef TO_FILL

    const rocgraph_handle_t*                        handle{};
    rocgraph_graph_t*                               graph{};
    const rocgraph_type_erased_device_array_view_t* source_vertices{};
    size_t                                          radius{};
    rocgraph_bool                                   do_expensive_check{};
    rocgraph_induced_subgraph_result_t**            result{};
    rocgraph_error_t**                              error{};
    auto                                            ret = rocgraph_extract_ego(
        handle, graph, source_vertices, radius, do_expensive_check, result, error);
    if(ret != rocgraph_status_success)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_internal_error);
    }
    CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
#endif
}

template <typename T>
void testing_rocgraph_extract_ego(const Arguments& arg)
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
    template void testing_rocgraph_extract_ego_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_rocgraph_extract_ego<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_extract_ego_extra(const Arguments& arg)
{
    testing_dispatch_extra(arg, "Egonet", Egonet, "EgonetNoWeights", EgonetNoWeights);
}
