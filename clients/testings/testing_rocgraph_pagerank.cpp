/*! \file*/

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

#include "rocgraph_clients_create_test_graph.hpp"
#include "rocgraph_test.hpp"

namespace
{

    template <typename weight_t, typename vertex_t>
    void generic_pagerank_test(vertex_t*     h_src,
                               vertex_t*     h_dst,
                               weight_t*     h_wgt,
                               weight_t*     h_result,
                               size_t        num_vertices,
                               size_t        num_edges,
                               rocgraph_bool store_transposed,
                               double        alpha,
                               double        epsilon,
                               size_t        max_iterations)
    {

        rocgraph_error_t* ret_error;

        rocgraph_handle_t*            p_handle = nullptr;
        rocgraph_graph_t*             p_graph  = nullptr;
        rocgraph_centrality_result_t* p_result = nullptr;

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

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_pagerank(p_handle,
                                                       p_graph,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       alpha,
                                                       epsilon,
                                                       max_iterations,
                                                       rocgraph_bool_false,
                                                       &p_result,
                                                       &ret_error),
                                     ret_error);

        rocgraph_type_erased_device_array_view_t* vertices;
        rocgraph_type_erased_device_array_view_t* pageranks;

        vertices  = rocgraph_centrality_result_get_vertices(p_result);
        pageranks = rocgraph_centrality_result_get_values(p_result);

        std::vector<vertex_t> h_vertices(num_vertices);
        std::vector<weight_t> h_pageranks(num_vertices);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_vertices.data(), vertices, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_pageranks.data(), pageranks, &ret_error),
            ret_error);

        ROCGRAPH_CLIENTS_EXPECT_INDIRECT_ARRAY_NEAR_TOLERANCE(num_vertices,
                                                              h_result,
                                                              1,
                                                              h_vertices.data(),
                                                              h_pageranks.data(),
                                                              1,
                                                              (const vertex_t*)nullptr,
                                                              0.001);

        rocgraph_centrality_result_free(p_result);
        rocgraph_sg_graph_free(p_graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    template <typename weight_t, typename vertex_t>
    void generic_pagerank_nonconverging_test(vertex_t*     h_src,
                                             vertex_t*     h_dst,
                                             weight_t*     h_wgt,
                                             weight_t*     h_result,
                                             size_t        num_vertices,
                                             size_t        num_edges,
                                             rocgraph_bool store_transposed,
                                             double        alpha,
                                             double        epsilon,
                                             size_t        max_iterations)
    {

        rocgraph_error_t* ret_error;

        rocgraph_handle_t*            p_handle = nullptr;
        rocgraph_graph_t*             p_graph  = nullptr;
        rocgraph_centrality_result_t* p_result = nullptr;

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

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_pagerank_allow_nonconvergence(p_handle,
                                                                            p_graph,
                                                                            nullptr,
                                                                            nullptr,
                                                                            nullptr,
                                                                            nullptr,
                                                                            alpha,
                                                                            epsilon,
                                                                            max_iterations,
                                                                            rocgraph_bool_false,
                                                                            &p_result,
                                                                            &ret_error),
                                     ret_error);

        rocgraph_type_erased_device_array_view_t* vertices;
        rocgraph_type_erased_device_array_view_t* pageranks;

        vertices  = rocgraph_centrality_result_get_vertices(p_result);
        pageranks = rocgraph_centrality_result_get_values(p_result);

        std::vector<vertex_t> h_vertices(num_vertices);
        std::vector<weight_t> h_pageranks(num_vertices);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_vertices.data(), vertices, &ret_error),
            ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_pageranks.data(), pageranks, &ret_error),
            ret_error);

        ROCGRAPH_CLIENTS_EXPECT_INDIRECT_ARRAY_NEAR_TOLERANCE(num_vertices,
                                                              h_result,
                                                              1,
                                                              h_vertices.data(),
                                                              h_pageranks.data(),
                                                              1,
                                                              (const vertex_t*)nullptr,
                                                              0.001);

        rocgraph_centrality_result_free(p_result);
        rocgraph_sg_graph_free(p_graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    template <typename weight_t, typename vertex_t>
    void generic_personalized_pagerank_test(vertex_t*     h_src,
                                            vertex_t*     h_dst,
                                            weight_t*     h_wgt,
                                            weight_t*     h_result,
                                            vertex_t*     h_personalization_vertices,
                                            weight_t*     h_personalization_values,
                                            size_t        num_vertices,
                                            size_t        num_edges,
                                            size_t        num_personalization_vertices,
                                            rocgraph_bool store_transposed,
                                            double        alpha,
                                            double        epsilon,
                                            size_t        max_iterations)
    {

        rocgraph_error_t* ret_error;

        rocgraph_handle_t*                        p_handle                      = nullptr;
        rocgraph_graph_t*                         p_graph                       = nullptr;
        rocgraph_centrality_result_t*             p_result                      = nullptr;
        rocgraph_type_erased_device_array_t*      personalization_vertices      = nullptr;
        rocgraph_type_erased_device_array_t*      personalization_values        = nullptr;
        rocgraph_type_erased_device_array_view_t* personalization_vertices_view = nullptr;
        rocgraph_type_erased_device_array_view_t* personalization_values_view   = nullptr;

        rocgraph_data_type_id vertex_tid = rocgraph_data_type_id_int32;
        rocgraph_data_type_id weight_tid = rocgraph_data_type_id_float32;

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
            rocgraph_type_erased_device_array_create(p_handle,
                                                     num_personalization_vertices,
                                                     vertex_tid,
                                                     &personalization_vertices,
                                                     &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_create(p_handle,
                                                     num_personalization_vertices,
                                                     weight_tid,
                                                     &personalization_values,
                                                     &ret_error),
            ret_error);

        personalization_vertices_view
            = rocgraph_type_erased_device_array_view(personalization_vertices);
        personalization_values_view
            = rocgraph_type_erased_device_array_view(personalization_values);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle,
                                         personalization_vertices_view,
                                         (rocgraph_byte_t*)h_personalization_vertices,
                                         &ret_error),
                                     ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle,
                                         personalization_values_view,
                                         (rocgraph_byte_t*)h_personalization_values,
                                         &ret_error),
                                     ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_personalized_pagerank(p_handle,
                                                                    p_graph,
                                                                    nullptr,
                                                                    nullptr,
                                                                    nullptr,
                                                                    nullptr,
                                                                    personalization_vertices_view,
                                                                    personalization_values_view,
                                                                    alpha,
                                                                    epsilon,
                                                                    max_iterations,
                                                                    rocgraph_bool_false,
                                                                    &p_result,
                                                                    &ret_error),
                                     ret_error);

        rocgraph_type_erased_device_array_view_t* vertices;
        rocgraph_type_erased_device_array_view_t* pageranks;

        vertices  = rocgraph_centrality_result_get_vertices(p_result);
        pageranks = rocgraph_centrality_result_get_values(p_result);

        std::vector<vertex_t> h_vertices(num_vertices);
        std::vector<weight_t> h_pageranks(num_vertices);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_vertices.data(), vertices, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_pageranks.data(), pageranks, &ret_error),
            ret_error);

        ROCGRAPH_CLIENTS_EXPECT_INDIRECT_ARRAY_NEAR_TOLERANCE(num_vertices,
                                                              h_result,
                                                              1,
                                                              h_vertices.data(),
                                                              h_pageranks.data(),
                                                              1,
                                                              (const vertex_t*)nullptr,
                                                              0.001);

        rocgraph_centrality_result_free(p_result);
        rocgraph_sg_graph_free(p_graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    template <typename weight_t, typename vertex_t>
    void generic_personalized_pagerank_nonconverging_test(vertex_t* h_src,
                                                          vertex_t* h_dst,
                                                          weight_t* h_wgt,
                                                          weight_t* h_result,
                                                          vertex_t* h_personalization_vertices,
                                                          weight_t* h_personalization_values,
                                                          size_t    num_vertices,
                                                          size_t    num_edges,
                                                          size_t    num_personalization_vertices,
                                                          rocgraph_bool store_transposed,
                                                          double        alpha,
                                                          double        epsilon,
                                                          size_t        max_iterations)
    {

        rocgraph_error_t* ret_error;

        rocgraph_handle_t*                        p_handle                      = nullptr;
        rocgraph_graph_t*                         p_graph                       = nullptr;
        rocgraph_centrality_result_t*             p_result                      = nullptr;
        rocgraph_type_erased_device_array_t*      personalization_vertices      = nullptr;
        rocgraph_type_erased_device_array_t*      personalization_values        = nullptr;
        rocgraph_type_erased_device_array_view_t* personalization_vertices_view = nullptr;
        rocgraph_type_erased_device_array_view_t* personalization_values_view   = nullptr;

        rocgraph_data_type_id vertex_tid = rocgraph_data_type_id_int32;
        rocgraph_data_type_id weight_tid = rocgraph_data_type_id_float32;

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
            rocgraph_type_erased_device_array_create(p_handle,
                                                     num_personalization_vertices,
                                                     vertex_tid,
                                                     &personalization_vertices,
                                                     &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_create(p_handle,
                                                     num_personalization_vertices,
                                                     weight_tid,
                                                     &personalization_values,
                                                     &ret_error),
            ret_error);

        personalization_vertices_view
            = rocgraph_type_erased_device_array_view(personalization_vertices);
        personalization_values_view
            = rocgraph_type_erased_device_array_view(personalization_values);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle,
                                         personalization_vertices_view,
                                         (rocgraph_byte_t*)h_personalization_vertices,
                                         &ret_error),
                                     ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle,
                                         personalization_values_view,
                                         (rocgraph_byte_t*)h_personalization_values,
                                         &ret_error),
                                     ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_personalized_pagerank_allow_nonconvergence(p_handle,
                                                                p_graph,
                                                                nullptr,
                                                                nullptr,
                                                                nullptr,
                                                                nullptr,
                                                                personalization_vertices_view,
                                                                personalization_values_view,
                                                                alpha,
                                                                epsilon,
                                                                max_iterations,
                                                                rocgraph_bool_false,
                                                                &p_result,
                                                                &ret_error),
            ret_error);

        rocgraph_type_erased_device_array_view_t* vertices;
        rocgraph_type_erased_device_array_view_t* pageranks;

        vertices  = rocgraph_centrality_result_get_vertices(p_result);
        pageranks = rocgraph_centrality_result_get_values(p_result);

        std::vector<vertex_t> h_vertices(num_vertices);
        std::vector<weight_t> h_pageranks(num_vertices);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_vertices.data(), vertices, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_pageranks.data(), pageranks, &ret_error),
            ret_error);

        ROCGRAPH_CLIENTS_EXPECT_INDIRECT_ARRAY_NEAR_TOLERANCE(num_vertices,
                                                              h_result,
                                                              1,
                                                              h_vertices.data(),
                                                              h_pageranks.data(),
                                                              1,
                                                              (const vertex_t*)nullptr,
                                                              0.001);

        rocgraph_centrality_result_free(p_result);
        rocgraph_sg_graph_free(p_graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    void Pagerank(const Arguments& arg)
    {
        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        weight_t h_result[] = {0.0915528, 0.168382, 0.0656831, 0.191468, 0.120677, 0.362237};

        double alpha          = 0.95;
        double epsilon        = 0.0001;
        size_t max_iterations = 20;

        // Pagerank wants store_transposed = rocgraph_bool_true
        generic_pagerank_test<weight_t, vertex_t>(h_src,
                                                  h_dst,
                                                  h_wgt,
                                                  h_result,
                                                  num_vertices,
                                                  num_edges,
                                                  rocgraph_bool_true,
                                                  alpha,
                                                  epsilon,
                                                  max_iterations);
    }

    void PagerankWithTranspose(const Arguments& arg)
    {
        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        weight_t h_result[] = {0.0915528, 0.168382, 0.0656831, 0.191468, 0.120677, 0.362237};

        double alpha          = 0.95;
        double epsilon        = 0.0001;
        size_t max_iterations = 20;

        // Pagerank wants store_transposed = rocgraph_bool_true
        //    This call will force rocgraph_pagerank to transpose the graph
        //    But we're passing src/dst backwards so the results will be the same
        generic_pagerank_test<weight_t, vertex_t>(h_src,
                                                  h_dst,
                                                  h_wgt,
                                                  h_result,
                                                  num_vertices,
                                                  num_edges,
                                                  rocgraph_bool_false,
                                                  alpha,
                                                  epsilon,
                                                  max_iterations);
    }

    void Pagerank4(const Arguments& arg)
    {
        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 3;
        size_t num_vertices = 4;

        vertex_t h_src[]    = {0, 1, 2};
        vertex_t h_dst[]    = {1, 2, 3};
        weight_t h_wgt[]    = {1.f, 1.f, 1.f};
        weight_t h_result[] = {
            0.11615584790706635f, 0.21488840878009796f, 0.29881080985069275f, 0.37014490365982056f};

        double alpha          = 0.85;
        double epsilon        = 1.0e-6;
        size_t max_iterations = 500;

        generic_pagerank_test<weight_t, vertex_t>(h_src,
                                                  h_dst,
                                                  h_wgt,
                                                  h_result,
                                                  num_vertices,
                                                  num_edges,
                                                  rocgraph_bool_false,
                                                  alpha,
                                                  epsilon,
                                                  max_iterations);
    }

    void Pagerank4WithTranspose(const Arguments& arg)
    {
        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 3;
        size_t num_vertices = 4;

        vertex_t h_src[]    = {0, 1, 2};
        vertex_t h_dst[]    = {1, 2, 3};
        weight_t h_wgt[]    = {1.f, 1.f, 1.f};
        weight_t h_result[] = {
            0.11615584790706635f, 0.21488840878009796f, 0.29881080985069275f, 0.37014490365982056f};

        double alpha          = 0.85;
        double epsilon        = 1.0e-6;
        size_t max_iterations = 500;

        generic_pagerank_test<weight_t, vertex_t>(h_src,
                                                  h_dst,
                                                  h_wgt,
                                                  h_result,
                                                  num_vertices,
                                                  num_edges,
                                                  rocgraph_bool_true,
                                                  alpha,
                                                  epsilon,
                                                  max_iterations);
    }

    void PagerankNonConvergence(const Arguments& arg)
    {
        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        weight_t h_result[] = {0.0776471, 0.167637, 0.0639699, 0.220202, 0.140046, 0.330498};

        double alpha          = 0.95;
        double epsilon        = 0.0001;
        size_t max_iterations = 2;

        // Pagerank wants store_transposed = rocgraph_bool_true
        generic_pagerank_nonconverging_test<weight_t, vertex_t>(h_src,
                                                                h_dst,
                                                                h_wgt,
                                                                h_result,
                                                                num_vertices,
                                                                num_edges,
                                                                rocgraph_bool_true,
                                                                alpha,
                                                                epsilon,
                                                                max_iterations);
    }

    void PersonalizedPagerank(const Arguments& arg)
    {
        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 3;
        size_t num_vertices = 4;

        vertex_t h_src[]    = {0, 1, 2};
        vertex_t h_dst[]    = {1, 2, 3};
        weight_t h_wgt[]    = {1.f, 1.f, 1.f};
        weight_t h_result[] = {0.0559233f, 0.159381f, 0.303244f, 0.481451f};

        vertex_t h_personalized_vertices[] = {0, 1, 2, 3};
        weight_t h_personalized_values[]   = {0.1, 0.2, 0.3, 0.4};

        double alpha          = 0.85;
        double epsilon        = 1.0e-6;
        size_t max_iterations = 500;

        generic_personalized_pagerank_test<weight_t, vertex_t>(h_src,
                                                               h_dst,
                                                               h_wgt,
                                                               h_result,
                                                               h_personalized_vertices,
                                                               h_personalized_values,
                                                               num_vertices,
                                                               num_edges,
                                                               num_vertices,
                                                               rocgraph_bool_false,
                                                               alpha,
                                                               epsilon,
                                                               max_iterations);
    }

    void PersonalizedPagerankNonConvergence(const Arguments& arg)
    {
        using vertex_t      = int32_t;
        using weight_t      = float;
        size_t num_edges    = 3;
        size_t num_vertices = 4;

        vertex_t h_src[]    = {0, 1, 2};
        vertex_t h_dst[]    = {1, 2, 3};
        weight_t h_wgt[]    = {1.f, 1.f, 1.f};
        weight_t h_result[] = {0.03625, 0.285, 0.32125, 0.3575};

        vertex_t h_personalized_vertices[] = {0, 1, 2, 3};
        weight_t h_personalized_values[]   = {0.1, 0.2, 0.3, 0.4};

        double alpha          = 0.85;
        double epsilon        = 1.0e-6;
        size_t max_iterations = 1;

        generic_personalized_pagerank_nonconverging_test<weight_t, vertex_t>(
            h_src,
            h_dst,
            h_wgt,
            h_result,
            h_personalized_vertices,
            h_personalized_values,
            num_vertices,
            num_edges,
            num_vertices,
            rocgraph_bool_false,
            alpha,
            epsilon,
            max_iterations);
    }

} // namespace
/*! \file */
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "rocgraph/rocgraph.h"
#include "testing.hpp"
#include "testing_rocgraph_pagerank.hpp"

template <typename T>
void testing_rocgraph_pagerank_bad_arg(const Arguments& arg)
{
#ifdef TO_FILL

    const rocgraph_handle_t*                        handle{};
    rocgraph_graph_t*                               graph{};
    const rocgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices{};
    const rocgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums{};
    const rocgraph_type_erased_device_array_view_t* initial_guess_vertices{};
    const rocgraph_type_erased_device_array_view_t* initial_guess_values{};
    double                                          alpha{};
    double                                          epsilon{};
    size_t                                          max_iterations{};
    rocgraph_bool                                   do_expensive_check{};
    rocgraph_centrality_result_t**                  result{};
    rocgraph_error_t**                              error{};
    auto                                            ret = rocgraph_pagerank(handle,
                                 graph,
                                 precomputed_vertex_out_weight_vertices,
                                 precomputed_vertex_out_weight_sums,
                                 initial_guess_vertices,
                                 initial_guess_values,
                                 alpha,
                                 epsilon,
                                 max_iterations,
                                 do_expensive_check,
                                 result,
                                 error);
    if(ret != rocgraph_status_success)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_internal_error);
    }
    CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
#endif
}

template <typename T>
void testing_rocgraph_pagerank(const Arguments& arg)
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
    template void testing_rocgraph_pagerank_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_rocgraph_pagerank<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_pagerank_extra(const Arguments& arg)
{
    testing_dispatch_extra(arg,
                           "Pagerank",
                           Pagerank,
                           "PagerankWithTranspose",
                           PagerankWithTranspose,
                           "Pagerank4",
                           Pagerank4,
                           "Pagerank4WithTranspose",
                           Pagerank4WithTranspose,
                           "PagerankNonConvergence",
                           PagerankNonConvergence,
                           "PersonalizedPagerank",
                           PersonalizedPagerank,
                           "PersonalizedPagerankNonConvergence",
                           PersonalizedPagerankNonConvergence);
}
