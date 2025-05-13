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

#include "testing_rocgraph_uniform_random_walks.hpp"
#include "rocgraph/rocgraph.h"
#include "rocgraph_clients_create_test_graph.hpp"
#include "rocgraph_clients_skip_test.hpp"
#include "rocgraph_test.hpp"
#include "testing.hpp"

namespace
{

    template <typename weight_t, typename vertex_t>
    void generic_uniform_random_walks_test(vertex_t*     h_src,
                                           vertex_t*     h_dst,
                                           weight_t*     h_wgt,
                                           size_t        num_vertices,
                                           size_t        num_edges,
                                           vertex_t*     h_start,
                                           size_t        num_starts,
                                           size_t        max_depth,
                                           rocgraph_bool renumber,
                                           rocgraph_bool store_transposed)
    {

        rocgraph_error_t* ret_error = nullptr;

        rocgraph_handle_t*             p_handle = nullptr;
        rocgraph_graph_t*              graph    = nullptr;
        rocgraph_random_walk_result_t* result   = nullptr;

        rocgraph_type_erased_device_array_t*      d_start      = nullptr;
        rocgraph_type_erased_device_array_view_t* d_start_view = nullptr;

        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        rocgraph_clients_create_test_graph(p_handle,
                                           h_src,
                                           h_dst,
                                           h_wgt,
                                           num_edges,
                                           store_transposed,
                                           renumber,
                                           rocgraph_bool_false,
                                           &graph,
                                           &ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_create(
                p_handle, num_starts, rocgraph_data_type_id_int32, &d_start, &ret_error),
            ret_error);

        d_start_view = rocgraph_type_erased_device_array_view(d_start);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_from_host(
                p_handle, d_start_view, (rocgraph_byte_t*)h_start, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_uniform_random_walks(
                p_handle, graph, d_start_view, max_depth, &result, &ret_error),
            ret_error);

        rocgraph_type_erased_device_array_view_t* verts;
        rocgraph_type_erased_device_array_view_t* wgts;

        verts = rocgraph_random_walk_result_get_paths(result);
        wgts  = rocgraph_random_walk_result_get_weights(result);

        size_t verts_size = rocgraph_type_erased_device_array_view_size(verts);
        size_t wgts_size  = rocgraph_type_erased_device_array_view_size(wgts);

        std::vector<vertex_t> h_result_verts(verts_size);
        std::vector<weight_t> h_result_wgts(wgts_size);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_verts.data(), verts, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_wgts.data(), wgts, &ret_error),
            ret_error);

        //  NOTE:  The C++ tester does a more thorough validation.  For our purposes
        //  here we will do a simpler validation, merely checking that all edges
        //  are actually part of the graph

        size_t unrenumbered_vertex_size = num_vertices;
        for(size_t i = 0; i < num_edges; ++i)
        {
            if(h_src[i] > unrenumbered_vertex_size)
                unrenumbered_vertex_size = h_src[i];
            if(h_dst[i] > unrenumbered_vertex_size)
                unrenumbered_vertex_size = h_dst[i];
        }
        ++unrenumbered_vertex_size;
        std::vector<weight_t> M(unrenumbered_vertex_size * unrenumbered_vertex_size, -1);

        for(size_t i = 0; i < num_edges; ++i)
            M[h_src[i] + unrenumbered_vertex_size * h_dst[i]] = h_wgt[i];

        ROCGRAPH_CLIENTS_EXPECT_EQ(rocgraph_random_walk_result_get_max_path_length(result),
                                   max_depth);

        for(int i = 0; i < num_starts; ++i)
        {
            ROCGRAPH_CLIENTS_EXPECT_EQ(h_start[i], h_result_verts[i * (max_depth + 1)]);
            for(size_t j = 0; j < max_depth; ++j)
            {
                int src_index = i * (max_depth + 1) + j;
                int dst_index = src_index + 1;
                if(h_result_verts[dst_index] < 0)
                {
                    if(h_result_verts[src_index] >= 0)
                    {
                        int departing_count = 0;
                        for(int k = 0; k < num_vertices; ++k)
                        {
                            if(M[h_result_verts[src_index] + unrenumbered_vertex_size * k] >= 0)
                                departing_count++;
                        }
                        ROCGRAPH_CLIENTS_EXPECT_EQ(departing_count, 0);
                    }
                }
                else
                {
                    ROCGRAPH_CLIENTS_EXPECT_EQ(
                        M[h_result_verts[src_index]
                          + unrenumbered_vertex_size * h_result_verts[dst_index]],
                        h_result_wgts[i * max_depth + j]);
                }
            }
        }

        rocgraph_random_walk_result_free(result);
        rocgraph_sg_graph_free(graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    template <typename weight_t, typename vertex_t>
    void generic_biased_random_walks_test(vertex_t*     h_src,
                                          vertex_t*     h_dst,
                                          weight_t*     h_wgt,
                                          size_t        num_vertices,
                                          size_t        num_edges,
                                          vertex_t*     h_start,
                                          size_t        num_starts,
                                          size_t        max_depth,
                                          rocgraph_bool renumber,
                                          rocgraph_bool store_transposed)
    {

        rocgraph_error_t* ret_error = nullptr;

        rocgraph_handle_t*             p_handle = nullptr;
        rocgraph_graph_t*              graph    = nullptr;
        rocgraph_random_walk_result_t* result   = nullptr;

        rocgraph_type_erased_device_array_t*      d_start      = nullptr;
        rocgraph_type_erased_device_array_view_t* d_start_view = nullptr;

        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        rocgraph_clients_create_test_graph(p_handle,
                                           h_src,
                                           h_dst,
                                           h_wgt,
                                           num_edges,
                                           store_transposed,
                                           renumber,
                                           rocgraph_bool_false,
                                           &graph,
                                           &ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_create(
                p_handle, num_starts, rocgraph_data_type_id_int32, &d_start, &ret_error),
            ret_error);

        d_start_view = rocgraph_type_erased_device_array_view(d_start);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_from_host(
                p_handle, d_start_view, (rocgraph_byte_t*)h_start, &ret_error),
            ret_error);

#if 1
        ROCGRAPH_CLIENTS_EXPECT_NE(
            rocgraph_biased_random_walks(
                p_handle, graph, d_start_view, max_depth, &result, &ret_error),
            rocgraph_status_success);
#else
        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_biased_random_walks(
                p_handle, graph, d_start_view, max_depth, &result, &ret_error),
            ret_error);

        rocgraph_type_erased_device_array_view_t* verts;
        rocgraph_type_erased_device_array_view_t* wgts;

        verts = rocgraph_random_walk_result_get_paths(result);
        wgts  = rocgraph_random_walk_result_get_weights(result);

        size_t verts_size = rocgraph_type_erased_device_array_view_size(verts);
        size_t wgts_size  = rocgraph_type_erased_device_array_view_size(wgts);

        std::vector<vertex_t> h_result_verts(verts_size);
        std::vector<vertex_t> h_result_wgts(wgts_size);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_verts.data(), verts, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_wgts.data(), wgts, &ret_error),
            ret_error);

        //  NOTE:  The C++ tester does a more thorough validation.  For our purposes
        //  here we will do a simpler validation, merely checking that all edges
        //  are actually part of the graph
        std::vector<weight_t> M(num_vertices * num_vertices, -1);

        for(int i = 0; i < num_edges; ++i)
            M[h_src[i] + num_vertices * h_dst[i]] = h_wgt[i];

        ROCGRAPH_CLIENTS_EXPECT_EQ(rocgraph_random_walk_result_get_max_path_length(), max_depth);

        for(int i = 0; i < num_starts; ++i)
        {
            ROCGRAPH_CLIENTS_EXPECT_EQ(
                M[h_start[i] + num_vertices * h_result_verts[i * (max_depth + 1)]],
                h_result_wgts[i * max_depth]);
            for(size_t j = 1; j < rocgraph_random_walk_result_get_max_path_length(); ++j)
                ROCGRAPH_CLIENTS_EXPECT_EQ(
                    M[h_start[i * (max_depth + 1) + j - 1]
                      + num_vertices * h_result_verts[i * (max_depth + 1) + j]],
                    h_result_wgts[i * max_depth + j - 1]);
        }

        rocgraph_random_walk_result_free(result);
#endif

        rocgraph_sg_graph_free(graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    template <typename weight_t, typename vertex_t>
    void generic_node2vec_random_walks_test(vertex_t*     h_src,
                                            vertex_t*     h_dst,
                                            weight_t*     h_wgt,
                                            size_t        num_vertices,
                                            size_t        num_edges,
                                            vertex_t*     h_start,
                                            size_t        num_starts,
                                            size_t        max_depth,
                                            weight_t      p,
                                            weight_t      q,
                                            rocgraph_bool renumber,
                                            rocgraph_bool store_transposed)
    {

        rocgraph_error_t* ret_error = nullptr;

        rocgraph_handle_t*             p_handle = nullptr;
        rocgraph_graph_t*              graph    = nullptr;
        rocgraph_random_walk_result_t* result   = nullptr;

        rocgraph_type_erased_device_array_t*      d_start      = nullptr;
        rocgraph_type_erased_device_array_view_t* d_start_view = nullptr;

        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        rocgraph_clients_create_test_graph(p_handle,
                                           h_src,
                                           h_dst,
                                           h_wgt,
                                           num_edges,
                                           store_transposed,
                                           renumber,
                                           rocgraph_bool_false,
                                           &graph,
                                           &ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_create(
                p_handle, num_starts, rocgraph_data_type_id_int32, &d_start, &ret_error),
            ret_error);

        d_start_view = rocgraph_type_erased_device_array_view(d_start);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_from_host(
                p_handle, d_start_view, (rocgraph_byte_t*)h_start, &ret_error),
            ret_error);

#if 1
        ROCGRAPH_CLIENTS_EXPECT_NE(
            rocgraph_node2vec_random_walks(
                p_handle, graph, d_start_view, max_depth, p, q, &result, &ret_error),
            rocgraph_status_success);
#else
        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_node2vec_random_walks(
                p_handle, graph, d_start_view, max_depth, p, q, &result, &ret_error),
            ret_error);

        rocgraph_type_erased_device_array_view_t* verts;
        rocgraph_type_erased_device_array_view_t* wgts;

        verts = rocgraph_random_walk_result_get_paths(result);
        wgts  = rocgraph_random_walk_result_get_weights(result);

        size_t verts_size = rocgraph_type_erased_device_array_view_size(verts);
        size_t wgts_size  = rocgraph_type_erased_device_array_view_size(wgts);

        std::vector<vertex_t> h_result_verts(verts_size);
        std::vector<vertex_t> h_result_wgts(wgts_size);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_verts.data(), verts, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_wgts.data(), wgts, &ret_error),
            ret_error);

        //  NOTE:  The C++ tester does a more thorough validation.  For our purposes
        //  here we will do a simpler validation, merely checking that all edges
        //  are actually part of the graph
        std::vector<weight_t> M(num_vertices * num_vertices, -1);

        for(int i = 0; i < num_edges; ++i)
            M[h_src[i] + num_vertices * h_dst[i]] = h_wgt[i];

        ROCGRAPH_CLIENTS_EXPECT_EQ(rocgraph_random_walk_result_get_max_path_length(), max_depth);

        for(int i = 0; i < num_starts; ++i)
        {
            ROCGRAPH_CLIENTS_EXPECT_EQ(
                M[h_start[i] + num_vertices * h_result_verts[i * (max_depth + 1)]],
                h_result_wgts[i * max_depth]);
            for(size_t j = 1; j < max_depth; ++j)
                ROCGRAPH_CLIENTS_EXPECT_EQ(
                    M[h_start[i * (max_depth + 1) + j - 1]
                      + num_vertices * h_result_verts[i * (max_depth + 1) + j]],
                    h_result_wgts[i * max_depth + j - 1]);
        }

        rocgraph_random_walk_result_free(result);
#endif

        rocgraph_sg_graph_free(graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    void UniformRandomWalks(const Arguments& arg)
    {
        using vertex_t = int32_t;
        using weight_t = float;
        ROCGRAPH_CLIENTS_SKIP_TEST(
            "Skipping UniformRandomWalks because ROCGRAPH_OPS is not supported in this release");
        size_t num_edges    = 8;
        size_t num_vertices = 6;
        size_t num_starts   = 2;

        vertex_t src[]   = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]   = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t wgt[]   = {0, 1, 2, 3, 4, 5, 6, 7};
        vertex_t start[] = {2, 2};

        generic_uniform_random_walks_test<weight_t, vertex_t>(src,
                                                              dst,
                                                              wgt,
                                                              num_vertices,
                                                              num_edges,
                                                              start,
                                                              num_starts,
                                                              3,
                                                              rocgraph_bool_false,
                                                              rocgraph_bool_false);
    }

    void BiasedRandomWalks(const Arguments& arg)
    {
        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 8;
        size_t num_vertices = 6;
        size_t num_starts   = 2;

        vertex_t src[]   = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]   = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t wgt[]   = {0, 1, 2, 3, 4, 5, 6, 7};
        vertex_t start[] = {2, 2};

        generic_biased_random_walks_test<weight_t, vertex_t>(src,
                                                             dst,
                                                             wgt,
                                                             num_vertices,
                                                             num_edges,
                                                             start,
                                                             num_starts,
                                                             3,
                                                             rocgraph_bool_false,
                                                             rocgraph_bool_false);
    }

    void Node2vecRandomWalks(const Arguments& arg)
    {
        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 8;
        size_t num_vertices = 6;
        size_t num_starts   = 2;

        vertex_t src[]   = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]   = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t wgt[]   = {0, 1, 2, 3, 4, 5, 6, 7};
        vertex_t start[] = {2, 2};
        weight_t p       = 5;
        weight_t q       = 9;

        generic_node2vec_random_walks_test<weight_t, vertex_t>(src,
                                                               dst,
                                                               wgt,
                                                               num_vertices,
                                                               num_edges,
                                                               start,
                                                               num_starts,
                                                               3,
                                                               p,
                                                               q,
                                                               rocgraph_bool_false,
                                                               rocgraph_bool_false);
    }

    void UniformRandomWalksOob(const Arguments& arg)
    {
        using vertex_t = int32_t;
        using weight_t = float;
        ROCGRAPH_CLIENTS_SKIP_TEST(
            "Skipping UniformRandomWalksOob because ROCGRAPH_OPS is not supported in this release");
        size_t num_edges    = 5;
        size_t num_vertices = 6;
        size_t num_starts   = 4;
        size_t max_depth    = 7;

        vertex_t src[]   = {1, 2, 4, 7, 3};
        vertex_t dst[]   = {5, 4, 1, 5, 2};
        weight_t wgt[]   = {0.4, 0.5, 0.6, 0.7, 0.8};
        vertex_t start[] = {2, 5, 3, 1};

        generic_uniform_random_walks_test<weight_t, vertex_t>(src,
                                                              dst,
                                                              wgt,
                                                              num_vertices,
                                                              num_edges,
                                                              start,
                                                              num_starts,
                                                              max_depth,
                                                              rocgraph_bool_true,
                                                              rocgraph_bool_false);
    }

} // namespace

template <typename T>
void testing_rocgraph_uniform_random_walks_bad_arg(const Arguments& arg)
{
#ifdef TO_FILL
    const rocgraph_handle_t*                        handle{};
    rocgraph_graph_t*                               graph{};
    const rocgraph_type_erased_device_array_view_t* start_vertices{};
    size_t                                          max_length{};
    rocgraph_random_walk_result_t**                 result{};
    rocgraph_error_t**                              error{};
    auto                                            ret
        = rocgraph_uniform_random_walks(handle, graph, start_vertices, max_length, result, error);
    if(ret != rocgraph_status_success)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_internal_error);
    }
    CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
#endif
}

template <typename T>
void testing_rocgraph_uniform_random_walks(const Arguments& arg)
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

#define INSTANTIATE(TYPE)                                                                    \
    template void testing_rocgraph_uniform_random_walks_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_rocgraph_uniform_random_walks<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_uniform_random_walks_extra(const Arguments& arg)
{
    testing_dispatch_extra(arg,
                           "UniformRandomWalks",
                           UniformRandomWalks,
                           "BiasedRandomWalks",
                           BiasedRandomWalks,
                           "Node2vecRandomWalks",
                           Node2vecRandomWalks,
                           "UniformRandomWalksOob",
                           UniformRandomWalksOob);
}
