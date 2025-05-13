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
#include "testing_rocgraph_node2vec.hpp"

namespace
{

    template <typename weight_t, typename edge_t, typename vertex_t>
    void generic_node2vec_test(vertex_t*     h_src,
                               vertex_t*     h_dst,
                               weight_t*     h_wgt,
                               vertex_t*     h_seeds,
                               size_t        num_vertices,
                               size_t        num_edges,
                               size_t        num_seeds,
                               size_t        max_depth,
                               rocgraph_bool compressed_result,
                               double        p,
                               double        q,
                               rocgraph_bool store_transposed)
    {

        rocgraph_error_t* ret_error = nullptr;

        rocgraph_handle_t*                        p_handle      = nullptr;
        rocgraph_graph_t*                         p_graph       = nullptr;
        rocgraph_random_walk_result_t*            p_result      = nullptr;
        rocgraph_type_erased_device_array_t*      p_sources     = nullptr;
        rocgraph_type_erased_device_array_view_t* p_source_view = nullptr;

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

        p_source_view = rocgraph_type_erased_device_array_view(p_sources);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_from_host(
                p_handle, p_source_view, (rocgraph_byte_t*)h_seeds, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_node2vec(p_handle,
                                                       p_graph,
                                                       p_source_view,
                                                       max_depth,
                                                       compressed_result,
                                                       p,
                                                       q,
                                                       &p_result,
                                                       &ret_error),
                                     ret_error);

        rocgraph_type_erased_device_array_view_t* paths;
        rocgraph_type_erased_device_array_view_t* path_sizes;
        rocgraph_type_erased_device_array_view_t* weights;
        size_t                                    max_path_length;

        max_path_length = rocgraph_random_walk_result_get_max_path_length(p_result);
        paths           = rocgraph_random_walk_result_get_paths(p_result);
        weights         = rocgraph_random_walk_result_get_weights(p_result);

        std::vector<vertex_t> h_paths(max_path_length * num_seeds);
        std::vector<weight_t> h_weights(max_path_length * num_seeds);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_paths.data(), paths, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_weights.data(), weights, &ret_error),
            ret_error);

        ROCGRAPH_CLIENTS_EXPECT_EQ(
            rocgraph_type_erased_device_array_view_size(paths),
            (rocgraph_type_erased_device_array_view_size(weights) + num_seeds));

        //  We can easily validate that the results of node2vec
        //  are feasible by converting the sparse (h_src,h_dst,h_wgt)
        //  into a dense host matrix and check each path.
        std::vector<weight_t> M(num_vertices * num_vertices, weight_t(0));

        for(int i = 0; i < num_edges; ++i)
            M[h_src[i] + num_vertices * h_dst[i]] = h_wgt[i];

        const weight_t EPSILON = 0.001;

        if(compressed_result)
        {
            path_sizes = rocgraph_random_walk_result_get_path_sizes(p_result);

            std::vector<edge_t> h_path_sizes(num_seeds);
            std::vector<edge_t> h_path_offsets(num_seeds + 1);

            CHECK_ROCGRAPH_SUCCESS_ERROR(
                rocgraph_type_erased_device_array_view_copy_to_host(
                    p_handle, (rocgraph_byte_t*)h_path_sizes.data(), path_sizes, &ret_error),
                ret_error);

            size_t path_size = 0;
            for(int i = 0; i < num_seeds; ++i)
                path_size += h_path_sizes[i];

            ROCGRAPH_CLIENTS_EXPECT_EQ(rocgraph_type_erased_device_array_view_size(paths),
                                       path_size);

            h_path_offsets[0] = 0;
            for(int i = 0; i < num_seeds; ++i)
                h_path_offsets[i + 1] = h_path_offsets[i] + h_path_sizes[i];

            for(int i = 0; i < num_seeds; ++i)
            {
                for(int j = h_path_offsets[i]; j < (h_path_offsets[i + 1] - 1); ++j)
                {
                    ROCGRAPH_CLIENTS_EXPECT_NEAR_TOLERANCE(
                        h_weights[j - i], M[h_paths[j] + num_vertices * h_paths[j + 1]], EPSILON);
                }
            }
        }
        else
        {
            for(int i = 0; i < num_seeds; ++i)
            {
                for(int j = 0; j < (max_path_length - 1); ++j)
                {
                    if(h_paths[i * max_path_length + j + 1] != num_vertices)
                    {
                        ROCGRAPH_CLIENTS_EXPECT_NEAR_TOLERANCE(
                            h_weights[i * (max_path_length - 1) + j],
                            M[h_paths[i * max_path_length + j]
                              + num_vertices * h_paths[i * max_path_length + j + 1]],
                            EPSILON);
                    }
                }
            }
        }
    }

    void Node2vec(const Arguments& arg)
    {

        using vertex_t      = int32_t;
        using edge_t        = int32_t;
        using weight_t      = float;
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t src[]     = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]     = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t wgt[]     = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t seeds[]   = {0, 0};
        size_t   max_depth = 4;

        generic_node2vec_test<weight_t, edge_t, vertex_t>(src,
                                                          dst,
                                                          wgt,
                                                          seeds,
                                                          num_vertices,
                                                          num_edges,
                                                          2,
                                                          max_depth,
                                                          rocgraph_bool_false,
                                                          0.8,
                                                          0.5,
                                                          rocgraph_bool_false);
    }

    void Node2vecShortDense(const Arguments& arg)
    {
        using vertex_t      = int32_t;
        using edge_t        = int32_t;
        using weight_t      = float;
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t src[]     = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]     = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t wgt[]     = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t seeds[]   = {2, 3};
        size_t   max_depth = 4;

        generic_node2vec_test<weight_t, edge_t, vertex_t>(src,
                                                          dst,
                                                          wgt,
                                                          seeds,
                                                          num_vertices,
                                                          num_edges,
                                                          2,
                                                          max_depth,
                                                          rocgraph_bool_false,
                                                          0.8,
                                                          0.5,
                                                          rocgraph_bool_false);
    }

    void Node2vecShortSparse(const Arguments& arg)
    {

        using vertex_t = int32_t;
        using edge_t   = int32_t;
        using weight_t = float;

        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t src[]     = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]     = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t wgt[]     = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t seeds[]   = {2, 3};
        size_t   max_depth = 4;

        // FIXME:  max_depth seems to be off by 1.  It's counting vertices
        //         instead of edges.
        generic_node2vec_test<weight_t, edge_t, vertex_t>(src,
                                                          dst,
                                                          wgt,
                                                          seeds,
                                                          num_vertices,
                                                          num_edges,
                                                          2,
                                                          max_depth,
                                                          rocgraph_bool_true,
                                                          0.8,
                                                          0.5,
                                                          rocgraph_bool_false);
    }

    void Node2vecKarate(const Arguments& arg)
    {
        using vertex_t      = int32_t;
        using edge_t        = int32_t;
        using weight_t      = float;
        size_t num_edges    = 156;
        size_t num_vertices = 34;

        vertex_t src[]
            = {1,  2,  3,  4,  5,  6,  7,  8,  10, 11, 12, 13, 17, 19, 21, 31, 2,  3,  7,  13,
               17, 19, 21, 30, 3,  7,  8,  9,  13, 27, 28, 32, 7,  12, 13, 6,  10, 6,  10, 16,
               16, 30, 32, 33, 33, 33, 32, 33, 32, 33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29,
               32, 33, 25, 27, 31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33, 0,  0,
               0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,
               1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  4,  4,  5,  5,  5,  6,  8,
               8,  8,  9,  13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22, 23, 23, 23, 23, 23,
               24, 24, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32};
        vertex_t dst[]
            = {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,
               1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  4,  4,  5,  5,  5,
               6,  8,  8,  8,  9,  13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22, 23, 23, 23,
               23, 23, 24, 24, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 1,  2,
               3,  4,  5,  6,  7,  8,  10, 11, 12, 13, 17, 19, 21, 31, 2,  3,  7,  13, 17, 19,
               21, 30, 3,  7,  8,  9,  13, 27, 28, 32, 7,  12, 13, 6,  10, 6,  10, 16, 16, 30,
               32, 33, 33, 33, 32, 33, 32, 33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29, 32, 33,
               25, 27, 31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33};
        weight_t wgt[]
            = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        vertex_t seeds[]   = {12, 28, 20, 23, 15, 26};
        size_t   max_depth = 5;

        generic_node2vec_test<weight_t, edge_t, vertex_t>(src,
                                                          dst,
                                                          wgt,
                                                          seeds,
                                                          num_vertices,
                                                          num_edges,
                                                          6,
                                                          max_depth,
                                                          rocgraph_bool_true,
                                                          0.8,
                                                          0.5,
                                                          rocgraph_bool_false);
    }

} // namespace

template <typename T>
void testing_rocgraph_node2vec_bad_arg(const Arguments& arg)
{
#ifdef TO_FILL

    const rocgraph_handle_t*                        handle{};
    rocgraph_graph_t*                               graph{};
    const rocgraph_type_erased_device_array_view_t* sources{};
    size_t                                          max_depth{};
    rocgraph_bool                                   compress_result{};
    double                                          p{};
    double                                          q{};
    rocgraph_random_walk_result_t**                 result{};
    rocgraph_error_t**                              error{};
    auto                                            ret = rocgraph_node2vec(
        handle, graph, sources, max_depth, compress_result, p, q, result, error);
    if(ret != rocgraph_status_success)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_internal_error);
    }
    CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
#endif
}

template <typename T>
void testing_rocgraph_node2vec(const Arguments& arg)
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

#define INSTANTIATE(TYPE)                                                        \
    template void testing_rocgraph_node2vec_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_rocgraph_node2vec<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_node2vec_extra(const Arguments& arg)
{
    testing_dispatch_extra(arg,
                           "Node2vec",
                           Node2vec,
                           "Node2vecShortDense",
                           Node2vecShortDense,
                           "Node2vecShortSparse",
                           Node2vecShortSparse,
                           "Node2vecKarate",
                           Node2vecKarate);
}
