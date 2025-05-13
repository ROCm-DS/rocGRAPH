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

typedef enum
{
    JACCARD,
    SORENSEN,
    OVERLAP
} similarity_t;

namespace
{
    template <typename weight_t, typename vertex_t>
    static void generic_similarity_test(vertex_t*     h_src,
                                        vertex_t*     h_dst,
                                        weight_t*     h_wgt,
                                        vertex_t*     h_first,
                                        vertex_t*     h_second,
                                        weight_t*     h_result,
                                        size_t        num_vertices,
                                        size_t        num_edges,
                                        size_t        num_pairs,
                                        rocgraph_bool store_transposed,
                                        rocgraph_bool use_weight,
                                        similarity_t  test_type)
    {
        rocgraph_data_type_id vertex_tid = rocgraph_data_type_id_int32;

        rocgraph_error_t* ret_error;

        rocgraph_handle_t*                        p_handle     = nullptr;
        rocgraph_graph_t*                         graph        = nullptr;
        rocgraph_similarity_result_t*             result       = nullptr;
        rocgraph_vertex_pairs_t*                  vertex_pairs = nullptr;
        rocgraph_type_erased_device_array_t*      v1           = nullptr;
        rocgraph_type_erased_device_array_t*      v2           = nullptr;
        rocgraph_type_erased_device_array_view_t* v1_view      = nullptr;
        rocgraph_type_erased_device_array_view_t* v2_view      = nullptr;

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

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_pairs, vertex_tid, &v1, &ret_error),
                                     ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_pairs, vertex_tid, &v2, &ret_error),
                                     ret_error);

        v1_view = rocgraph_type_erased_device_array_view(v1);
        v2_view = rocgraph_type_erased_device_array_view(v2);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle, v1_view, (rocgraph_byte_t*)h_first, &ret_error),
                                     ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle, v2_view, (rocgraph_byte_t*)h_second, &ret_error),
                                     ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_create_vertex_pairs(
                p_handle, graph, v1_view, v2_view, &vertex_pairs, &ret_error),
            ret_error);

        switch(test_type)
        {
        case JACCARD:
            CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_jaccard_coefficients(p_handle,
                                                                       graph,
                                                                       vertex_pairs,
                                                                       use_weight,
                                                                       rocgraph_bool_false,
                                                                       &result,
                                                                       &ret_error),
                                         ret_error);
            break;
        case SORENSEN:
            CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_sorensen_coefficients(p_handle,
                                                                        graph,
                                                                        vertex_pairs,
                                                                        use_weight,
                                                                        rocgraph_bool_false,
                                                                        &result,
                                                                        &ret_error),
                                         ret_error);
            break;
        case OVERLAP:
            CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_overlap_coefficients(p_handle,
                                                                       graph,
                                                                       vertex_pairs,
                                                                       use_weight,
                                                                       rocgraph_bool_false,
                                                                       &result,
                                                                       &ret_error),
                                         ret_error);
            break;
        }

        rocgraph_type_erased_device_array_view_t* similarity_coefficient;

        similarity_coefficient = rocgraph_similarity_result_get_similarity(result);

        std::vector<weight_t> h_similarity_coefficient(num_pairs);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_to_host(
                                         p_handle,
                                         (rocgraph_byte_t*)h_similarity_coefficient.data(),
                                         similarity_coefficient,
                                         &ret_error),
                                     ret_error);

        ROCGRAPH_CLIENTS_EXPECT_SEGMENT_NEAR_TOLERANCE(
            num_pairs, h_similarity_coefficient.data(), h_result, 0.001);

        if(result != nullptr)
            rocgraph_similarity_result_free(result);
        if(vertex_pairs != nullptr)
            rocgraph_vertex_pairs_free(vertex_pairs);
        rocgraph_sg_graph_free(graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    template <typename weight_t, typename vertex_t>
    static void generic_all_pairs_similarity_test(vertex_t*     h_src,
                                                  vertex_t*     h_dst,
                                                  weight_t*     h_wgt,
                                                  vertex_t*     h_first,
                                                  vertex_t*     h_second,
                                                  weight_t*     h_result,
                                                  size_t        num_vertices,
                                                  size_t        num_edges,
                                                  size_t        num_pairs,
                                                  rocgraph_bool store_transposed,
                                                  rocgraph_bool use_weight,
                                                  size_t        topk,
                                                  similarity_t  test_type)
    {

        rocgraph_error_t* ret_error;

        rocgraph_handle_t*                        p_handle      = nullptr;
        rocgraph_graph_t*                         graph         = nullptr;
        rocgraph_similarity_result_t*             result        = nullptr;
        rocgraph_type_erased_device_array_view_t* vertices_view = nullptr;

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

        switch(test_type)
        {
        case JACCARD:
            CHECK_ROCGRAPH_SUCCESS_ERROR(
                rocgraph_all_pairs_jaccard_coefficients(p_handle,
                                                        graph,
                                                        vertices_view,
                                                        use_weight,
                                                        topk,
                                                        rocgraph_bool_false,
                                                        &result,
                                                        &ret_error),
                ret_error);
            break;
        case SORENSEN:
            CHECK_ROCGRAPH_SUCCESS_ERROR(
                rocgraph_all_pairs_sorensen_coefficients(p_handle,
                                                         graph,
                                                         vertices_view,
                                                         use_weight,
                                                         topk,
                                                         rocgraph_bool_false,
                                                         &result,
                                                         &ret_error),
                ret_error);
            break;
        case OVERLAP:
            CHECK_ROCGRAPH_SUCCESS_ERROR(
                rocgraph_all_pairs_overlap_coefficients(p_handle,
                                                        graph,
                                                        vertices_view,
                                                        use_weight,
                                                        topk,
                                                        rocgraph_bool_false,
                                                        &result,
                                                        &ret_error),
                ret_error);
            break;
        }

        rocgraph_type_erased_device_array_view_t* similarity_coefficient;

        rocgraph_vertex_pairs_t* vertex_pairs;
        vertex_pairs           = rocgraph_similarity_result_get_vertex_pairs(result);
        similarity_coefficient = rocgraph_similarity_result_get_similarity(result);

        rocgraph_type_erased_device_array_view_t* result_v1;
        rocgraph_type_erased_device_array_view_t* result_v2;

        result_v1               = rocgraph_vertex_pairs_get_first(vertex_pairs);
        result_v2               = rocgraph_vertex_pairs_get_second(vertex_pairs);
        size_t result_num_pairs = rocgraph_type_erased_device_array_view_size(result_v1);

        ROCGRAPH_CLIENTS_EXPECT_EQ(result_num_pairs, num_pairs);

        std::vector<vertex_t> h_result_v1(result_num_pairs);
        std::vector<vertex_t> h_result_v2(result_num_pairs);
        std::vector<weight_t> h_similarity_coefficient(result_num_pairs);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_v1.data(), result_v1, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_v2.data(), result_v2, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_to_host(
                                         p_handle,
                                         (rocgraph_byte_t*)h_similarity_coefficient.data(),
                                         similarity_coefficient,
                                         &ret_error),
                                     ret_error);

        std::vector<weight_t> result_matrix(num_vertices * num_vertices, 0);

        for(int i = 0; i < num_pairs; ++i)
            result_matrix[h_result_v1[i] + num_vertices * h_result_v2[i]]
                = h_similarity_coefficient[i];

        for(int i = 0; i < num_pairs; ++i)
        {
            ROCGRAPH_CLIENTS_EXPECT_NEAR_TOLERANCE(
                result_matrix[h_first[i] + num_vertices * h_second[i]], h_result[i], 0.001);
        }

        if(result != nullptr)
            rocgraph_similarity_result_free(result);
        rocgraph_sg_graph_free(graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

} // namespace
