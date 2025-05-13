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
#include "testing_rocgraph_extract_induced_subgraph.hpp"

/*
 * Simple check of creating a graph from a COO on device memory.
 */
namespace
{
    template <typename weight_t, typename vertex_t>
    static void generic_induced_subgraph_test(vertex_t*     h_src,
                                              vertex_t*     h_dst,
                                              weight_t*     h_wgt,
                                              size_t        num_vertices,
                                              size_t        num_edges,
                                              rocgraph_bool store_transposed,
                                              size_t*       h_subgraph_offsets,
                                              vertex_t*     h_subgraph_vertices,
                                              size_t        num_subgraph_offsets,
                                              vertex_t*     h_result_src,
                                              vertex_t*     h_result_dst,
                                              weight_t*     h_result_wgt,
                                              size_t*       h_result_offsets,
                                              size_t        num_results)
    {
        rocgraph_error_t* ret_error;

        rocgraph_handle_t*                        p_handle               = nullptr;
        rocgraph_graph_t*                         graph                  = nullptr;
        rocgraph_type_erased_device_array_t*      subgraph_offsets       = nullptr;
        rocgraph_type_erased_device_array_t*      subgraph_vertices      = nullptr;
        rocgraph_type_erased_device_array_view_t* subgraph_offsets_view  = nullptr;
        rocgraph_type_erased_device_array_view_t* subgraph_vertices_view = nullptr;

        rocgraph_induced_subgraph_result_t* result = nullptr;

        rocgraph_data_type_id vertex_tid = rocgraph_data_type_id_int32;
        rocgraph_data_type_id size_t_tid = rocgraph_data_type_id_size_t;

        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));
        rocgraph_clients_create_test_graph(p_handle,
                                           h_src,
                                           h_dst,
                                           h_wgt,
                                           num_edges,
                                           store_transposed,
                                           rocgraph_bool_false,
                                           rocgraph_bool_false,
                                           &graph,
                                           &ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_create(
                p_handle, num_subgraph_offsets, size_t_tid, &subgraph_offsets, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_create(p_handle,
                                                     h_subgraph_offsets[num_subgraph_offsets - 1],
                                                     vertex_tid,
                                                     &subgraph_vertices,
                                                     &ret_error),
            ret_error);

        subgraph_offsets_view  = rocgraph_type_erased_device_array_view(subgraph_offsets);
        subgraph_vertices_view = rocgraph_type_erased_device_array_view(subgraph_vertices);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_from_host(
                p_handle, subgraph_offsets_view, (rocgraph_byte_t*)h_subgraph_offsets, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle,
                                         subgraph_vertices_view,
                                         (rocgraph_byte_t*)h_subgraph_vertices,
                                         &ret_error),
                                     ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_extract_induced_subgraph(p_handle,
                                                                       graph,
                                                                       subgraph_offsets_view,
                                                                       subgraph_vertices_view,
                                                                       rocgraph_bool_false,
                                                                       &result,
                                                                       &ret_error),
                                     ret_error);

        rocgraph_type_erased_device_array_view_t* extracted_src;
        rocgraph_type_erased_device_array_view_t* extracted_dst;
        rocgraph_type_erased_device_array_view_t* extracted_wgt;
        rocgraph_type_erased_device_array_view_t* extracted_graph_offsets;

        extracted_src           = rocgraph_induced_subgraph_get_sources(result);
        extracted_dst           = rocgraph_induced_subgraph_get_destinations(result);
        extracted_wgt           = rocgraph_induced_subgraph_get_edge_weights(result);
        extracted_graph_offsets = rocgraph_induced_subgraph_get_subgraph_offsets(result);

        size_t extracted_size = rocgraph_type_erased_device_array_view_size(extracted_src);

        std::vector<vertex_t> h_extracted_src(extracted_size);
        std::vector<vertex_t> h_extracted_dst(extracted_size);
        std::vector<weight_t> h_extracted_wgt(extracted_size);
        std::vector<size_t>   h_extracted_graph_offsets(num_subgraph_offsets);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_extracted_src.data(), extracted_src, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_extracted_dst.data(), extracted_dst, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_extracted_wgt.data(), extracted_wgt, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_to_host(
                                         p_handle,
                                         (rocgraph_byte_t*)h_extracted_graph_offsets.data(),
                                         extracted_graph_offsets,
                                         &ret_error),
                                     ret_error);

        ROCGRAPH_CLIENTS_EXPECT_EQ(extracted_size, num_results);

        ROCGRAPH_CLIENTS_EXPECT_SEGMENT_EQ(
            num_subgraph_offsets, h_extracted_graph_offsets.data(), h_result_offsets);

        for(size_t i = 0; i < num_results; ++i)
        {
            bool found = false;
            for(size_t j = 0; (j < num_results) && !found; ++j)
            {
                found = ((h_extracted_src[i] == h_result_src[j])
                         && (h_extracted_dst[i] == h_result_dst[j])
                         && rocgraph_clients_are_near_tolerance(
                             h_extracted_wgt[i], h_result_wgt[j], 0.001));
            }
            ROCGRAPH_CLIENTS_EXPECT_TRUE(found);
        }
    }

    void InducedSubgraph(const Arguments& arg)
    {
        using vertex_t              = int32_t;
        using weight_t              = float;
        size_t num_edges            = 8;
        size_t num_vertices         = 6;
        size_t num_subgraph_offsets = 2;
        size_t num_results          = 5;

        vertex_t h_src[]               = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]               = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]               = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        size_t   h_subgraph_offsets[]  = {0, 4};
        vertex_t h_subgraph_vertices[] = {0, 1, 2, 3};
        vertex_t h_result_src[]        = {0, 1, 2, 2, 2};
        vertex_t h_result_dst[]        = {1, 3, 0, 1, 3};
        weight_t h_result_wgt[]        = {0.1f, 2.1f, 5.1f, 3.1f, 4.1f};
        size_t   h_result_offsets[]    = {0, 5};

        generic_induced_subgraph_test(h_src,
                                      h_dst,
                                      h_wgt,
                                      num_vertices,
                                      num_edges,
                                      rocgraph_bool_false,
                                      h_subgraph_offsets,
                                      h_subgraph_vertices,
                                      num_subgraph_offsets,
                                      h_result_src,
                                      h_result_dst,
                                      h_result_wgt,
                                      h_result_offsets,
                                      num_results);
    }

} // namespace

template <typename T>
void testing_rocgraph_extract_induced_subgraph_bad_arg(const Arguments& arg)
{
#ifdef TO_FILL

    const rocgraph_handle_t*                        handle{};
    rocgraph_graph_t*                               graph{};
    const rocgraph_type_erased_device_array_view_t* subgraph_offsets{};
    const rocgraph_type_erased_device_array_view_t* subgraph_vertices{};
    rocgraph_bool                                   do_expensive_check{};
    rocgraph_induced_subgraph_result_t**            result{};
    rocgraph_error_t**                              error{};
    auto                                            ret = rocgraph_extract_induced_subgraph(
        handle, graph, subgraph_offsets, subgraph_vertices, do_expensive_check, result, error);
    if(ret != rocgraph_status_success)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_internal_error);
    }
    CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
#endif
}

template <typename T>
void testing_rocgraph_extract_induced_subgraph(const Arguments& arg)
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

#define INSTANTIATE(TYPE)                                                                        \
    template void testing_rocgraph_extract_induced_subgraph_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_rocgraph_extract_induced_subgraph<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_extract_induced_subgraph_extra(const Arguments& arg)
{
    testing_dispatch_extra(arg, "InducedSubgraph", InducedSubgraph);
}
