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

#include "testing_rocgraph_sg_graph_create.hpp"
#include "rocgraph/rocgraph.h"
#include "rocgraph_clients_create_test_graph.hpp"
#include "rocgraph_clients_skip_test.hpp"
#include "rocgraph_test.hpp"
#include "testing.hpp"

#include <cstdio>

/*
 * Simple check of creating a graph from a COO on device memory.
 */

namespace
{

    void CreateSgGraphSimple(const Arguments& arg)
    {
        using vertex_t = int32_t;
        using weight_t = float;

        rocgraph_error_t*       ret_error;
        static constexpr size_t num_edges = 8;
        // static constexpr size_t                num_vertices = 6;

        vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[] = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};

        rocgraph_handle_t*          p_handle = nullptr;
        rocgraph_graph_t*           graph    = nullptr;
        rocgraph_graph_properties_t properties;

        properties.is_symmetric  = rocgraph_bool_false;
        properties.is_multigraph = rocgraph_bool_false;

        rocgraph_data_type_id vertex_tid = rocgraph_data_type_id_int32;
        // rocgraph_data_type_id edge_tid   = rocgraph_data_type_id_int32;
        rocgraph_data_type_id weight_tid = rocgraph_data_type_id_float32;

        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        rocgraph_type_erased_device_array_t*      src;
        rocgraph_type_erased_device_array_t*      dst;
        rocgraph_type_erased_device_array_t*      wgt;
        rocgraph_type_erased_device_array_view_t* src_view;
        rocgraph_type_erased_device_array_view_t* dst_view;
        rocgraph_type_erased_device_array_view_t* wgt_view;

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_edges, vertex_tid, &src, &ret_error),
                                     ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_edges, vertex_tid, &dst, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_edges, weight_tid, &wgt, &ret_error),
                                     ret_error);

        src_view = rocgraph_type_erased_device_array_view(src);
        dst_view = rocgraph_type_erased_device_array_view(dst);
        wgt_view = rocgraph_type_erased_device_array_view(wgt);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle, src_view, (rocgraph_byte_t*)h_src, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle, dst_view, (rocgraph_byte_t*)h_dst, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle, wgt_view, (rocgraph_byte_t*)h_wgt, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_graph_create_sg(p_handle,
                                                              &properties,
                                                              nullptr,
                                                              src_view,
                                                              dst_view,
                                                              wgt_view,
                                                              nullptr,
                                                              nullptr,
                                                              rocgraph_bool_false,
                                                              rocgraph_bool_false,
                                                              rocgraph_bool_false,
                                                              rocgraph_bool_false,
                                                              rocgraph_bool_false,
                                                              &graph,
                                                              &ret_error),
                                     ret_error);
        rocgraph_graph_free(graph);

        rocgraph_type_erased_device_array_view_free(wgt_view);
        rocgraph_type_erased_device_array_view_free(dst_view);
        rocgraph_type_erased_device_array_view_free(src_view);
        rocgraph_type_erased_device_array_free(wgt);
        rocgraph_type_erased_device_array_free(dst);
        rocgraph_type_erased_device_array_free(src);

        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    void generate_topology_loop(int32_t nvertices,
                                int32_t* __restrict__ h_src,
                                int32_t* __restrict__ h_dst)
    {
        for(int32_t i = 0; i < nvertices; ++i)
        {
            h_src[i] = i;
            h_dst[i] = (i + 1) % nvertices;
        }
    }

    void generate_topology_star(int32_t nvertices,
                                int32_t* __restrict__ h_src,
                                int32_t* __restrict__ h_dst)
    {
        int32_t nedges = nvertices - 1;
        for(int32_t i = 0; i < nedges; ++i)
            h_src[i] = 0;
        for(int32_t i = 0; i < nedges; ++i)
            h_dst[i] = i + 1;
    }

    template <typename T>
    void CreateSgGraphSimple_main(const Arguments& arg)
    {
        int32_t M      = arg.M;
        using vertex_t = int32_t;

        rocgraph_error_t* ret_error;
        size_t            num_edges = M;

        std::vector<vertex_t> v_h_src(num_edges);
        std::vector<vertex_t> v_h_dst(num_edges);
        std::vector<T>        v_h_wgt(num_edges, 1);
        if(1)
            generate_topology_loop(M, v_h_src.data(), v_h_dst.data());
        else
            generate_topology_star(M, v_h_src.data(), v_h_dst.data());
        vertex_t* h_src = (vertex_t*)v_h_src.data();
        vertex_t* h_dst = (vertex_t*)v_h_dst.data();
        T*        h_wgt = (T*)v_h_wgt.data();

        rocgraph_handle_t*          p_handle = nullptr;
        rocgraph_graph_t*           graph    = nullptr;
        rocgraph_graph_properties_t properties;

        properties.is_symmetric  = rocgraph_bool_false;
        properties.is_multigraph = rocgraph_bool_false;

        rocgraph_data_type_id vertex_tid = rocgraph_data_type_id_int32;
        // rocgraph_data_type_id edge_tid   = rocgraph_data_type_id_int32;
        rocgraph_data_type_id weight_tid = rocgraph_data_type_id_float32;

        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        rocgraph_type_erased_device_array_t*      src;
        rocgraph_type_erased_device_array_t*      dst;
        rocgraph_type_erased_device_array_t*      wgt;
        rocgraph_type_erased_device_array_view_t* src_view;
        rocgraph_type_erased_device_array_view_t* dst_view;
        rocgraph_type_erased_device_array_view_t* wgt_view;

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_edges, vertex_tid, &src, &ret_error),
                                     ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_edges, vertex_tid, &dst, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_edges, weight_tid, &wgt, &ret_error),
                                     ret_error);

        src_view = rocgraph_type_erased_device_array_view(src);
        dst_view = rocgraph_type_erased_device_array_view(dst);
        wgt_view = rocgraph_type_erased_device_array_view(wgt);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle, src_view, (rocgraph_byte_t*)h_src, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle, dst_view, (rocgraph_byte_t*)h_dst, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle, wgt_view, (rocgraph_byte_t*)h_wgt, &ret_error),
                                     ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_graph_create_sg(p_handle,
                                                              &properties,
                                                              nullptr,
                                                              src_view,
                                                              dst_view,
                                                              wgt_view,
                                                              nullptr,
                                                              nullptr,
                                                              rocgraph_bool_false,
                                                              rocgraph_bool_false,
                                                              rocgraph_bool_false,
                                                              rocgraph_bool_false,
                                                              rocgraph_bool_false,
                                                              &graph,
                                                              &ret_error),
                                     ret_error);

        rocgraph_graph_free(graph);

        rocgraph_type_erased_device_array_view_free(wgt_view);
        rocgraph_type_erased_device_array_view_free(dst_view);
        rocgraph_type_erased_device_array_view_free(src_view);
        rocgraph_type_erased_device_array_free(wgt);
        rocgraph_type_erased_device_array_free(dst);
        rocgraph_type_erased_device_array_free(src);

        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    void CreateSgGraphCsr(const Arguments& arg)
    {
        using vertex_t = int32_t;
        using edge_t   = int32_t;
        using weight_t = float;
        ROCGRAPH_CLIENTS_SKIP_TEST(
            "Skipping CreateSgGraphCsr because ROCGRAPH_OPS is not supported in this release");
        rocgraph_error_t*       ret_error;
        static constexpr size_t num_edges    = 8;
        static constexpr size_t num_vertices = 6;

        /*
      vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4};
      vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5};
    */
        edge_t   h_offsets[] = {0, 1, 3, 6, 7, 8, 8};
        vertex_t h_indices[] = {1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_start[]   = {0, 1, 2, 3, 4, 5};
        weight_t h_wgt[]     = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};

        rocgraph_bool                   with_replacement = rocgraph_bool_false;
        rocgraph_bool                   return_hops      = rocgraph_bool_true;
        rocgraph_prior_sources_behavior prior_sources_behavior
            = rocgraph_prior_sources_behavior_default;
        rocgraph_bool             dedupe_sources   = rocgraph_bool_false;
        rocgraph_bool             renumber_results = rocgraph_bool_false;
        rocgraph_compression_type compression      = rocgraph_compression_type_coo;
        rocgraph_bool             compress_per_hop = rocgraph_bool_false;

        rocgraph_handle_t*          p_handle = nullptr;
        rocgraph_graph_t*           graph    = nullptr;
        rocgraph_graph_properties_t properties;

        properties.is_symmetric  = rocgraph_bool_false;
        properties.is_multigraph = rocgraph_bool_false;

        rocgraph_data_type_id vertex_tid = rocgraph_data_type_id_int32;
        // rocgraph_data_type_id edge_tid   = rocgraph_data_type_id_int32;
        rocgraph_data_type_id weight_tid = rocgraph_data_type_id_float32;

        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        rocgraph_type_erased_device_array_t*      offsets;
        rocgraph_type_erased_device_array_t*      indices;
        rocgraph_type_erased_device_array_t*      wgt;
        rocgraph_type_erased_device_array_view_t* offsets_view;
        rocgraph_type_erased_device_array_view_t* indices_view;
        rocgraph_type_erased_device_array_view_t* wgt_view;

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_create(
                p_handle, num_vertices + 1, vertex_tid, &offsets, &ret_error),
            ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_edges, vertex_tid, &indices, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_edges, weight_tid, &wgt, &ret_error),
                                     ret_error);
        offsets_view = rocgraph_type_erased_device_array_view(offsets);
        indices_view = rocgraph_type_erased_device_array_view(indices);
        wgt_view     = rocgraph_type_erased_device_array_view(wgt);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_from_host(
                p_handle, offsets_view, (rocgraph_byte_t*)h_offsets, &ret_error),
            ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_from_host(
                p_handle, indices_view, (rocgraph_byte_t*)h_indices, &ret_error),
            ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle, wgt_view, (rocgraph_byte_t*)h_wgt, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_sg_graph_create_from_csr(p_handle,
                                                                       &properties,
                                                                       offsets_view,
                                                                       indices_view,
                                                                       wgt_view,
                                                                       nullptr,
                                                                       nullptr,
                                                                       rocgraph_bool_false,
                                                                       rocgraph_bool_false,
                                                                       rocgraph_bool_false,
                                                                       &graph,
                                                                       &ret_error),
                                     ret_error);
        std::vector<weight_t> M(num_vertices * num_vertices, -1);

        for(int i = 0; i < num_vertices; ++i)
            for(size_t j = h_offsets[i]; j < h_offsets[i + 1]; ++j)
            {
                M[i + num_vertices * h_indices[j]] = h_wgt[j];
            }

        int fan_out[] = {-1};

        rocgraph_type_erased_device_array_t*      d_start        = nullptr;
        rocgraph_type_erased_device_array_view_t* d_start_view   = nullptr;
        rocgraph_type_erased_host_array_view_t*   h_fan_out_view = nullptr;
        rocgraph_sample_result_t*                 result         = nullptr;

        h_fan_out_view
            = rocgraph_type_erased_host_array_view_create(fan_out, 1, rocgraph_data_type_id_int32);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_create(
                p_handle, num_vertices, rocgraph_data_type_id_int32, &d_start, &ret_error),
            ret_error);
        d_start_view = rocgraph_type_erased_device_array_view(d_start);
        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_from_host(
                p_handle, d_start_view, (rocgraph_byte_t*)h_start, &ret_error),
            ret_error);

        rocgraph_rng_state_t* rng_state;
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_rng_state_create(p_handle, 0, &rng_state, &ret_error),
                                     ret_error);
        rocgraph_sampling_options_t* sampling_options;

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_sampling_options_create(&sampling_options, &ret_error), ret_error);
        rocgraph_sampling_set_with_replacement(sampling_options, with_replacement);
        rocgraph_sampling_set_return_hops(sampling_options, return_hops);
        rocgraph_sampling_set_prior_sources_behavior(sampling_options, prior_sources_behavior);
        rocgraph_sampling_set_dedupe_sources(sampling_options, dedupe_sources);
        rocgraph_sampling_set_renumber_results(sampling_options, renumber_results);
        rocgraph_sampling_set_compression_type(sampling_options, compression);
        rocgraph_sampling_set_compress_per_hop(sampling_options, compress_per_hop);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_uniform_neighbor_sample(p_handle,
                                                                      graph,
                                                                      d_start_view,
                                                                      nullptr,
                                                                      nullptr,
                                                                      nullptr,
                                                                      nullptr,
                                                                      h_fan_out_view,
                                                                      rng_state,
                                                                      sampling_options,
                                                                      rocgraph_bool_false,
                                                                      &result,
                                                                      &ret_error),
                                     ret_error);

        rocgraph_type_erased_device_array_view_t* srcs;
        rocgraph_type_erased_device_array_view_t* dsts;
        rocgraph_type_erased_device_array_view_t* wgts;

        srcs = rocgraph_sample_result_get_sources(result);
        dsts = rocgraph_sample_result_get_destinations(result);
        wgts = rocgraph_sample_result_get_edge_weight(result);

        size_t result_size = rocgraph_type_erased_device_array_view_size(srcs);

        std::vector<vertex_t> h_result_srcs(result_size);
        std::vector<vertex_t> h_result_dsts(result_size);
        std::vector<weight_t> h_result_wgts(result_size);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_srcs.data(), srcs, &ret_error),
            ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_dsts.data(), dsts, &ret_error),
            ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_wgts.data(), wgts, &ret_error),
            ret_error);

        ROCGRAPH_CLIENTS_EXPECT_EQ(result_size, num_edges);

        for(size_t i = 0; i < result_size; ++i)
        {
            ROCGRAPH_CLIENTS_EXPECT_EQ(M[h_result_srcs[i] + num_vertices * h_result_dsts[i]],
                                       h_result_wgts[i]);
        }

        rocgraph_sample_result_free(result);
        rocgraph_graph_free(graph);
        rocgraph_type_erased_device_array_view_free(wgt_view);
        rocgraph_type_erased_device_array_view_free(indices_view);
        rocgraph_type_erased_device_array_view_free(offsets_view);
        rocgraph_type_erased_device_array_free(wgt);
        rocgraph_type_erased_device_array_free(indices);
        rocgraph_type_erased_device_array_free(offsets);

        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
        rocgraph_sampling_options_free(sampling_options);
    }

    void CreateSgGraphSymmetricError(const Arguments& arg)
    {
        using vertex_t = int32_t;
        using weight_t = float;
        rocgraph_error_t*       ret_error;
        static constexpr size_t num_edges = 8;
        // static constexpr size_t                num_vertices = 6;

        vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[] = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};

        rocgraph_handle_t*          p_handle = nullptr;
        rocgraph_graph_t*           graph    = nullptr;
        rocgraph_graph_properties_t properties;

        properties.is_symmetric  = rocgraph_bool_true;
        properties.is_multigraph = rocgraph_bool_false;

        rocgraph_data_type_id vertex_tid = rocgraph_data_type_id_int32;
        // rocgraph_data_type_id edge_tid   = rocgraph_data_type_id_int32;
        rocgraph_data_type_id weight_tid = rocgraph_data_type_id_float32;

        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        rocgraph_type_erased_device_array_t*      src;
        rocgraph_type_erased_device_array_t*      dst;
        rocgraph_type_erased_device_array_t*      wgt;
        rocgraph_type_erased_device_array_view_t* src_view;
        rocgraph_type_erased_device_array_view_t* dst_view;
        rocgraph_type_erased_device_array_view_t* wgt_view;

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_edges, vertex_tid, &src, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_edges, vertex_tid, &dst, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_edges, weight_tid, &wgt, &ret_error),
                                     ret_error);
        src_view = rocgraph_type_erased_device_array_view(src);
        dst_view = rocgraph_type_erased_device_array_view(dst);
        wgt_view = rocgraph_type_erased_device_array_view(wgt);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle, src_view, (rocgraph_byte_t*)h_src, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle, dst_view, (rocgraph_byte_t*)h_dst, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle, wgt_view, (rocgraph_byte_t*)h_wgt, &ret_error),
                                     ret_error);

        rocgraph_status ret_code;
        ret_code = rocgraph_graph_create_sg(p_handle,
                                            &properties,
                                            nullptr,
                                            src_view,
                                            dst_view,
                                            wgt_view,
                                            nullptr,
                                            nullptr,
                                            rocgraph_bool_false,
                                            rocgraph_bool_false,
                                            rocgraph_bool_false,
                                            rocgraph_bool_false,
                                            rocgraph_bool_true,
                                            &graph,
                                            &ret_error);

        ROCGRAPH_CLIENTS_EXPECT_NE(ret_code, rocgraph_status_success);

        if(ret_code == rocgraph_status_success)
        {
            rocgraph_graph_free(graph);
        }

        rocgraph_type_erased_device_array_view_free(wgt_view);
        rocgraph_type_erased_device_array_view_free(dst_view);
        rocgraph_type_erased_device_array_view_free(src_view);
        rocgraph_type_erased_device_array_free(wgt);
        rocgraph_type_erased_device_array_free(dst);
        rocgraph_type_erased_device_array_free(src);

        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    void CreateSgGraphWithIsolatedVertices(const Arguments& arg)
    {
        using vertex_t = int32_t;
        using weight_t = float;
        rocgraph_error_t*       ret_error;
        static constexpr size_t num_edges      = 8;
        static constexpr size_t num_vertices   = 7;
        double                  alpha          = 0.95;
        double                  epsilon        = 0.0001;
        static constexpr size_t max_iterations = 20;

        vertex_t h_vertices[] = {0, 1, 2, 3, 4, 5, 6};
        vertex_t h_src[]      = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]      = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]      = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        weight_t h_result[]
            = {0.0859168, 0.158029, 0.0616337, 0.179675, 0.113239, 0.339873, 0.0616337};

        rocgraph_handle_t*          p_handle = nullptr;
        rocgraph_graph_t*           graph    = nullptr;
        rocgraph_graph_properties_t properties;

        properties.is_symmetric  = rocgraph_bool_false;
        properties.is_multigraph = rocgraph_bool_false;

        rocgraph_data_type_id vertex_tid = rocgraph_data_type_id_int32;
        // rocgraph_data_type_id edge_tid   = rocgraph_data_type_id_int32;
        rocgraph_data_type_id weight_tid = rocgraph_data_type_id_float32;

        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        rocgraph_type_erased_device_array_t*      vertices;
        rocgraph_type_erased_device_array_t*      src;
        rocgraph_type_erased_device_array_t*      dst;
        rocgraph_type_erased_device_array_t*      wgt;
        rocgraph_type_erased_device_array_view_t* vertices_view;
        rocgraph_type_erased_device_array_view_t* src_view;
        rocgraph_type_erased_device_array_view_t* dst_view;
        rocgraph_type_erased_device_array_view_t* wgt_view;

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_vertices, vertex_tid, &vertices, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_edges, vertex_tid, &src, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_edges, vertex_tid, &dst, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_edges, weight_tid, &wgt, &ret_error),
                                     ret_error);
        vertices_view = rocgraph_type_erased_device_array_view(vertices);
        src_view      = rocgraph_type_erased_device_array_view(src);
        dst_view      = rocgraph_type_erased_device_array_view(dst);
        wgt_view      = rocgraph_type_erased_device_array_view(wgt);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_from_host(
                p_handle, vertices_view, (rocgraph_byte_t*)h_vertices, &ret_error),
            ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle, src_view, (rocgraph_byte_t*)h_src, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle, dst_view, (rocgraph_byte_t*)h_dst, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle, wgt_view, (rocgraph_byte_t*)h_wgt, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_graph_create_sg(p_handle,
                                                              &properties,
                                                              vertices_view,
                                                              src_view,
                                                              dst_view,
                                                              wgt_view,
                                                              nullptr,
                                                              nullptr,
                                                              rocgraph_bool_false,
                                                              rocgraph_bool_false,
                                                              rocgraph_bool_false,
                                                              rocgraph_bool_false,
                                                              rocgraph_bool_false,
                                                              &graph,
                                                              &ret_error),
                                     ret_error);
        rocgraph_centrality_result_t* result = nullptr;

        // To verify we will call pagerank
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_pagerank(p_handle,
                                                       graph,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       alpha,
                                                       epsilon,
                                                       max_iterations,
                                                       rocgraph_bool_false,
                                                       &result,
                                                       &ret_error),
                                     ret_error);
        rocgraph_type_erased_device_array_view_t* result_vertices;
        rocgraph_type_erased_device_array_view_t* pageranks;

        result_vertices = rocgraph_centrality_result_get_vertices(result);
        pageranks       = rocgraph_centrality_result_get_values(result);

        vertex_t h_result_vertices[num_vertices];
        weight_t h_pageranks[num_vertices];

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_vertices, result_vertices, &ret_error),
            ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_pageranks, pageranks, &ret_error),
            ret_error);

        ROCGRAPH_CLIENTS_EXPECT_INDIRECT_ARRAY_NEAR_TOLERANCE(num_vertices,
                                                              h_result,
                                                              1,
                                                              h_result_vertices,
                                                              h_pageranks,
                                                              1,
                                                              (const vertex_t*)nullptr,
                                                              0.001);

        rocgraph_centrality_result_free(result);
        rocgraph_graph_free(graph);

        rocgraph_type_erased_device_array_view_free(wgt_view);
        rocgraph_type_erased_device_array_view_free(dst_view);
        rocgraph_type_erased_device_array_view_free(src_view);
        rocgraph_type_erased_device_array_view_free(vertices_view);
        rocgraph_type_erased_device_array_free(wgt);
        rocgraph_type_erased_device_array_free(dst);
        rocgraph_type_erased_device_array_free(src);
        rocgraph_type_erased_device_array_free(vertices);

        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    void CreateSgGraphCsrWithIsolated(const Arguments& arg)
    {
        using vertex_t = int32_t;
        using edge_t   = int32_t;
        using weight_t = float;
        rocgraph_error_t*       ret_error;
        static constexpr size_t num_edges      = 8;
        static constexpr size_t num_vertices   = 7;
        double                  alpha          = 0.95;
        double                  epsilon        = 0.0001;
        static constexpr size_t max_iterations = 20;

        /*
	  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4};
	  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5};
	*/
        edge_t   h_offsets[] = {0, 1, 3, 6, 7, 8, 8, 8};
        vertex_t h_indices[] = {1, 3, 4, 0, 1, 3, 5, 5};
        // vertex_t h_start[]   = {0, 1, 2, 3, 4, 5};
        weight_t h_wgt[] = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        weight_t h_result[]
            = {0.0859168, 0.158029, 0.0616337, 0.179675, 0.113239, 0.339873, 0.0616337};

        rocgraph_handle_t*          p_handle = nullptr;
        rocgraph_graph_t*           graph    = nullptr;
        rocgraph_graph_properties_t properties;

        properties.is_symmetric  = rocgraph_bool_false;
        properties.is_multigraph = rocgraph_bool_false;

        rocgraph_data_type_id vertex_tid = rocgraph_data_type_id_int32;
        // rocgraph_data_type_id edge_tid   = rocgraph_data_type_id_int32;
        rocgraph_data_type_id weight_tid = rocgraph_data_type_id_float32;

        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        rocgraph_type_erased_device_array_t*      offsets;
        rocgraph_type_erased_device_array_t*      indices;
        rocgraph_type_erased_device_array_t*      wgt;
        rocgraph_type_erased_device_array_view_t* offsets_view;
        rocgraph_type_erased_device_array_view_t* indices_view;
        rocgraph_type_erased_device_array_view_t* wgt_view;

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_create(
                p_handle, num_vertices + 1, vertex_tid, &offsets, &ret_error),
            ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_edges, vertex_tid, &indices, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_edges, weight_tid, &wgt, &ret_error),
                                     ret_error);
        offsets_view = rocgraph_type_erased_device_array_view(offsets);
        indices_view = rocgraph_type_erased_device_array_view(indices);
        wgt_view     = rocgraph_type_erased_device_array_view(wgt);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_from_host(
                p_handle, offsets_view, (rocgraph_byte_t*)h_offsets, &ret_error),
            ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_from_host(
                p_handle, indices_view, (rocgraph_byte_t*)h_indices, &ret_error),
            ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle, wgt_view, (rocgraph_byte_t*)h_wgt, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_sg_graph_create_from_csr(p_handle,
                                                                       &properties,
                                                                       offsets_view,
                                                                       indices_view,
                                                                       wgt_view,
                                                                       nullptr,
                                                                       nullptr,
                                                                       rocgraph_bool_false,
                                                                       rocgraph_bool_false,
                                                                       rocgraph_bool_false,
                                                                       &graph,
                                                                       &ret_error),
                                     ret_error);
        rocgraph_centrality_result_t* result = nullptr;

        // To verify we will call pagerank
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_pagerank(p_handle,
                                                       graph,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       alpha,
                                                       epsilon,
                                                       max_iterations,
                                                       rocgraph_bool_false,
                                                       &result,
                                                       &ret_error),
                                     ret_error);
        rocgraph_type_erased_device_array_view_t* result_vertices;
        rocgraph_type_erased_device_array_view_t* pageranks;

        result_vertices = rocgraph_centrality_result_get_vertices(result);
        pageranks       = rocgraph_centrality_result_get_values(result);

        vertex_t h_result_vertices[num_vertices];
        weight_t h_pageranks[num_vertices];

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_vertices, result_vertices, &ret_error),
            ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_pageranks, pageranks, &ret_error),
            ret_error);

        ROCGRAPH_CLIENTS_EXPECT_INDIRECT_ARRAY_NEAR_TOLERANCE(num_vertices,
                                                              h_result,
                                                              1,
                                                              h_result_vertices,
                                                              h_pageranks,
                                                              1,
                                                              (const vertex_t*)nullptr,
                                                              0.001);

        rocgraph_centrality_result_free(result);
        rocgraph_graph_free(graph);
        rocgraph_type_erased_device_array_view_free(wgt_view);
        rocgraph_type_erased_device_array_view_free(indices_view);
        rocgraph_type_erased_device_array_view_free(offsets_view);
        rocgraph_type_erased_device_array_free(wgt);
        rocgraph_type_erased_device_array_free(indices);
        rocgraph_type_erased_device_array_free(offsets);

        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    void CreateSgGraphWithIsolatedVerticesMultiInput(const Arguments& arg)
    {
        using vertex_t = int32_t;
        using weight_t = float;
        rocgraph_error_t*       ret_error;
        static constexpr size_t num_edges      = 66;
        static constexpr size_t num_vertices   = 7;
        double                  alpha          = 0.95;
        double                  epsilon        = 0.0001;
        static constexpr size_t max_iterations = 20;

        vertex_t h_vertices[] = {0, 1, 2, 3, 4, 5, 6};
        vertex_t h_src[]      = {0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5, 0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5,
                                 0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5, 0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5,
                                 0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5, 0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5};
        vertex_t h_dst[]      = {1, 3, 4, 0, 1, 3, 5, 5, 5, 5, 5, 1, 3, 4, 0, 1, 3, 5, 5, 5, 5, 5,
                                 1, 3, 4, 0, 1, 3, 5, 5, 5, 5, 5, 1, 3, 4, 0, 1, 3, 5, 5, 5, 5, 5,
                                 1, 3, 4, 0, 1, 3, 5, 5, 5, 5, 5, 1, 3, 4, 0, 1, 3, 5, 5, 5, 5, 5};
        weight_t h_wgt[]
            = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 3.2f, 3.2f, 1.7f, 0.1f, 2.1f, 1.1f,
               5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 3.2f, 3.2f, 1.7f, 0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f,
               7.2f, 3.2f, 3.2f, 3.2f, 1.7f, 0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 3.2f,
               3.2f, 1.7f, 0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 3.2f, 3.2f, 1.7f, 0.1f,
               2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 3.2f, 3.2f, 1.7f};
        weight_t h_result[]
            = {0.0859168, 0.158029, 0.0616337, 0.179675, 0.113239, 0.339873, 0.0616337};

        rocgraph_handle_t*          p_handle = nullptr;
        rocgraph_graph_t*           graph    = nullptr;
        rocgraph_graph_properties_t properties;

        properties.is_symmetric  = rocgraph_bool_false;
        properties.is_multigraph = rocgraph_bool_false;

        rocgraph_data_type_id vertex_tid = rocgraph_data_type_id_int32;
        // rocgraph_data_type_id edge_tid   = rocgraph_data_type_id_int32;
        rocgraph_data_type_id weight_tid = rocgraph_data_type_id_float32;

        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        rocgraph_type_erased_device_array_t*      vertices;
        rocgraph_type_erased_device_array_t*      src;
        rocgraph_type_erased_device_array_t*      dst;
        rocgraph_type_erased_device_array_t*      wgt;
        rocgraph_type_erased_device_array_view_t* vertices_view;
        rocgraph_type_erased_device_array_view_t* src_view;
        rocgraph_type_erased_device_array_view_t* dst_view;
        rocgraph_type_erased_device_array_view_t* wgt_view;

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_vertices, vertex_tid, &vertices, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_edges, vertex_tid, &src, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_edges, vertex_tid, &dst, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_edges, weight_tid, &wgt, &ret_error),
                                     ret_error);
        vertices_view = rocgraph_type_erased_device_array_view(vertices);
        src_view      = rocgraph_type_erased_device_array_view(src);
        dst_view      = rocgraph_type_erased_device_array_view(dst);
        wgt_view      = rocgraph_type_erased_device_array_view(wgt);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_from_host(
                p_handle, vertices_view, (rocgraph_byte_t*)h_vertices, &ret_error),
            ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle, src_view, (rocgraph_byte_t*)h_src, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle, dst_view, (rocgraph_byte_t*)h_dst, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle, wgt_view, (rocgraph_byte_t*)h_wgt, &ret_error),
                                     ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_graph_create_sg(p_handle,
                                                              &properties,
                                                              vertices_view,
                                                              src_view,
                                                              dst_view,
                                                              wgt_view,
                                                              nullptr,
                                                              nullptr,
                                                              rocgraph_bool_false,
                                                              rocgraph_bool_false,
                                                              rocgraph_bool_true,
                                                              rocgraph_bool_true,
                                                              rocgraph_bool_false,
                                                              &graph,
                                                              &ret_error),
                                     ret_error);
        rocgraph_centrality_result_t* result = nullptr;

        // To verify we will call pagerank
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_pagerank(p_handle,
                                                       graph,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       alpha,
                                                       epsilon,
                                                       max_iterations,
                                                       rocgraph_bool_false,
                                                       &result,
                                                       &ret_error),
                                     ret_error);
        rocgraph_type_erased_device_array_view_t* result_vertices;
        rocgraph_type_erased_device_array_view_t* pageranks;

        result_vertices = rocgraph_centrality_result_get_vertices(result);
        pageranks       = rocgraph_centrality_result_get_values(result);

        vertex_t h_result_vertices[num_vertices];
        weight_t h_pageranks[num_vertices];

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_vertices, result_vertices, &ret_error),
            ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_pageranks, pageranks, &ret_error),
            ret_error);

        ROCGRAPH_CLIENTS_EXPECT_INDIRECT_ARRAY_NEAR_TOLERANCE(num_vertices,
                                                              h_result,
                                                              1,
                                                              h_result_vertices,
                                                              h_pageranks,
                                                              1,
                                                              (const vertex_t*)nullptr,
                                                              0.001);

        rocgraph_centrality_result_free(result);
        rocgraph_graph_free(graph);

        rocgraph_type_erased_device_array_view_free(wgt_view);
        rocgraph_type_erased_device_array_view_free(dst_view);
        rocgraph_type_erased_device_array_view_free(src_view);
        rocgraph_type_erased_device_array_view_free(vertices_view);
        rocgraph_type_erased_device_array_free(wgt);
        rocgraph_type_erased_device_array_free(dst);
        rocgraph_type_erased_device_array_free(src);
        rocgraph_type_erased_device_array_free(vertices);

        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

} // namespace

template <typename T>
void testing_rocgraph_sg_graph_create_bad_arg(const Arguments& arg)
{
#ifdef TO_FILL
    const rocgraph_handle_t*                        handle{};
    const rocgraph_graph_properties_t*              properties{};
    const rocgraph_type_erased_device_array_view_t* src{};
    const rocgraph_type_erased_device_array_view_t* dst{};
    const rocgraph_type_erased_device_array_view_t* weights{};
    const rocgraph_type_erased_device_array_view_t* edge_ids{};
    const rocgraph_type_erased_device_array_view_t* edge_type_ids{};
    rocgraph_bool                                   store_transposed{};
    rocgraph_bool                                   renumber{};
    rocgraph_bool                                   do_expensive_check{};
    rocgraph_graph_t**                              graph{};
    rocgraph_error_t**                              error{};
    auto                                            ret = rocgraph_sg_graph_create(handle,
                                        properties,
                                        src,
                                        dst,
                                        weights,
                                        edge_ids,
                                        edge_type_ids,
                                        store_transposed,
                                        renumber,
                                        do_expensive_check,
                                        graph,
                                        error);
    if(ret != rocgraph_status_success)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_internal_error);
    }
    CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
#else

#endif
}

template <typename T>
void testing_rocgraph_sg_graph_create(const Arguments& arg)
{
    CreateSgGraphSimple_main<T>(arg);
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
#else
#endif
}

#define INSTANTIATE(TYPE)                                                               \
    template void testing_rocgraph_sg_graph_create_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_rocgraph_sg_graph_create<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_sg_graph_create_extra(const Arguments& arg)
{
    testing_dispatch_extra(arg,
                           "CreateSgGraphSimple",
                           CreateSgGraphSimple,
                           "CreateSgGraphCsr",
                           CreateSgGraphCsr,
                           "CreateSgGraphSymmetricError",
                           CreateSgGraphSymmetricError,
                           "CreateSgGraphWithIsolatedVertices",
                           CreateSgGraphWithIsolatedVertices,
                           "CreateSgGraphCsrWithIsolated",
                           CreateSgGraphCsrWithIsolated,
                           "CreateSgGraphWithIsolatedVerticesMultiInput",
                           CreateSgGraphWithIsolatedVerticesMultiInput);
}
