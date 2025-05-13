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
#include "testing_rocgraph_uniform_neighbor_sample.hpp"
#include <stdlib.h>

#include "rocgraph_clients_skip_test.hpp"

rocgraph_data_type_id vertex_tid    = rocgraph_data_type_id_int32;
rocgraph_data_type_id edge_tid      = rocgraph_data_type_id_int32;
rocgraph_data_type_id weight_tid    = rocgraph_data_type_id_float32;
rocgraph_data_type_id edge_id_tid   = rocgraph_data_type_id_int32;
rocgraph_data_type_id edge_type_tid = rocgraph_data_type_id_int32;

namespace
{
    template <typename vertex_t>
    int vertex_id_compare_function(const void* a, const void* b)
    {
        if(*((vertex_t*)a) < *((vertex_t*)b))
            return -1;
        else if(*((vertex_t*)a) > *((vertex_t*)b))
            return 1;
        else
            return 0;
    }

    template <typename weight_t, typename edge_t, typename vertex_t>
    void
        generic_uniform_neighbor_sample_test(vertex_t*                       h_src,
                                             vertex_t*                       h_dst,
                                             weight_t*                       h_wgt,
                                             edge_t*                         h_edge_ids,
                                             int32_t*                        h_edge_types,
                                             size_t                          num_vertices,
                                             size_t                          num_edges,
                                             vertex_t*                       h_start,
                                             int*                            h_start_labels,
                                             size_t                          num_start_vertices,
                                             size_t                          num_start_labels,
                                             int*                            fan_out,
                                             size_t                          fan_out_size,
                                             rocgraph_bool                   with_replacement,
                                             rocgraph_bool                   return_hops,
                                             rocgraph_prior_sources_behavior prior_sources_behavior,
                                             rocgraph_bool                   dedupe_sources,
                                             rocgraph_bool                   renumber_results)
    {
        // Create graph
        rocgraph_error_t*         ret_error = nullptr;
        rocgraph_graph_t*         graph     = nullptr;
        rocgraph_sample_result_t* result    = nullptr;

        rocgraph_handle_t* p_handle = nullptr;
        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        rocgraph_clients_create_sg_test_graph(p_handle,
                                              vertex_tid,
                                              edge_tid,
                                              h_src,
                                              h_dst,
                                              weight_tid,
                                              h_wgt,
                                              edge_type_tid,
                                              h_edge_types,
                                              edge_id_tid,
                                              h_edge_ids,
                                              num_edges,
                                              rocgraph_bool_false,
                                              rocgraph_bool_true,
                                              rocgraph_bool_false,
                                              rocgraph_bool_false,
                                              &graph,
                                              &ret_error);

        rocgraph_type_erased_device_array_t*      d_start             = nullptr;
        rocgraph_type_erased_device_array_view_t* d_start_view        = nullptr;
        rocgraph_type_erased_device_array_t*      d_start_labels      = nullptr;
        rocgraph_type_erased_device_array_view_t* d_start_labels_view = nullptr;
        rocgraph_type_erased_host_array_view_t*   h_fan_out_view      = nullptr;

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_create(
                p_handle, num_start_vertices, rocgraph_data_type_id_int32, &d_start, &ret_error),
            ret_error);

        d_start_view = rocgraph_type_erased_device_array_view(d_start);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_from_host(
                p_handle, d_start_view, (rocgraph_byte_t*)h_start, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_create(p_handle,
                                                     num_start_vertices,
                                                     rocgraph_data_type_id_int32,
                                                     &d_start_labels,
                                                     &ret_error),
            ret_error);

        d_start_labels_view = rocgraph_type_erased_device_array_view(d_start_labels);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_from_host(
                p_handle, d_start_labels_view, (rocgraph_byte_t*)h_start_labels, &ret_error),
            ret_error);

        h_fan_out_view = rocgraph_type_erased_host_array_view_create(
            fan_out, fan_out_size, rocgraph_data_type_id_int32);

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

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_uniform_neighbor_sample(p_handle,
                                                                      graph,
                                                                      d_start_view,
                                                                      d_start_labels_view,
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

        rocgraph_sampling_options_free(sampling_options);

        rocgraph_type_erased_device_array_view_t* result_srcs;
        rocgraph_type_erased_device_array_view_t* result_dsts;
        rocgraph_type_erased_device_array_view_t* result_edge_id;
        rocgraph_type_erased_device_array_view_t* result_weights;
        rocgraph_type_erased_device_array_view_t* result_edge_types;
        rocgraph_type_erased_device_array_view_t* result_hops;
        rocgraph_type_erased_device_array_view_t* result_offsets;
        rocgraph_type_erased_device_array_view_t* result_labels;
        rocgraph_type_erased_device_array_view_t* result_renumber_map;
        rocgraph_type_erased_device_array_view_t* result_renumber_map_offsets;

        result_srcs                 = rocgraph_sample_result_get_sources(result);
        result_dsts                 = rocgraph_sample_result_get_destinations(result);
        result_edge_id              = rocgraph_sample_result_get_edge_id(result);
        result_weights              = rocgraph_sample_result_get_edge_weight(result);
        result_edge_types           = rocgraph_sample_result_get_edge_type(result);
        result_hops                 = rocgraph_sample_result_get_hop(result);
        result_hops                 = rocgraph_sample_result_get_hop(result);
        result_offsets              = rocgraph_sample_result_get_offsets(result);
        result_labels               = rocgraph_sample_result_get_start_labels(result);
        result_renumber_map         = rocgraph_sample_result_get_renumber_map(result);
        result_renumber_map_offsets = rocgraph_sample_result_get_renumber_map_offsets(result);

        size_t result_size         = rocgraph_type_erased_device_array_view_size(result_srcs);
        size_t result_offsets_size = rocgraph_type_erased_device_array_view_size(result_offsets);
        size_t renumber_map_size   = 0;

        if(renumber_results)
        {
            renumber_map_size = rocgraph_type_erased_device_array_view_size(result_renumber_map);
        }

        std::vector<vertex_t> h_result_srcs(result_size);
        std::vector<vertex_t> h_result_dsts(result_size);
        std::vector<edge_t>   h_result_edge_id(result_size);
        std::vector<weight_t> h_result_weight(result_size);
        std::vector<int32_t>  h_result_edge_types(result_size);
        std::vector<int32_t>  h_result_hops(result_size);
        std::vector<size_t>   h_result_offsets(result_offsets_size);
        std::vector<int>      h_result_labels(num_start_labels);
        std::vector<vertex_t> h_renumber_map(renumber_map_size);
        std::vector<size_t>   h_renumber_map_offsets(result_offsets_size);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_srcs.data(), result_srcs, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_dsts.data(), result_dsts, &ret_error),
            ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_edge_id.data(), result_edge_id, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_weight.data(), result_weights, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_to_host(
                                         p_handle,
                                         (rocgraph_byte_t*)h_result_edge_types.data(),
                                         result_edge_types,
                                         &ret_error),
                                     ret_error);

        ROCGRAPH_CLIENTS_EXPECT_EQ(result_hops, nullptr);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_offsets.data(), result_offsets, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_labels.data(), result_labels, &ret_error),
            ret_error);

        for(int k = 0; k < result_offsets_size - 1; k += fan_out_size)
        {
            for(int h = 0; h < fan_out_size; ++h)
            {
                int hop_start = h_result_offsets[k + h];
                int hop_end   = h_result_offsets[k + h + 1];
                for(int i = hop_start; i < hop_end; ++i)
                {
                    h_result_hops[i] = h;
                }
            }
        }

        for(int k = 0; k < num_start_labels + 1; ++k)
        {
            h_result_offsets[k] = h_result_offsets[k * fan_out_size];
        }
        result_offsets_size = num_start_labels + 1;

        if(renumber_results)
        {
            CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_to_host(
                                             p_handle,
                                             (rocgraph_byte_t*)h_renumber_map.data(),
                                             result_renumber_map,
                                             &ret_error),
                                         ret_error);

            CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_to_host(
                                             p_handle,
                                             (rocgraph_byte_t*)h_renumber_map_offsets.data(),
                                             result_renumber_map_offsets,
                                             &ret_error),
                                         ret_error);
        }

        //  First, check that all edges are actually part of the graph
        std::vector<weight_t> M_w(num_vertices * num_vertices);
        std::vector<edge_t>   M_edge_id(num_vertices * num_vertices);
        std::vector<int32_t>  M_edge_type(num_vertices * num_vertices);

        for(int i = 0; i < num_vertices; ++i)
            for(int j = 0; j < num_vertices; ++j)
            {
                M_w[i + num_vertices * j]         = 0.0;
                M_edge_id[i + num_vertices * j]   = -1;
                M_edge_type[i + num_vertices * j] = -1;
            }

        for(int i = 0; i < num_edges; ++i)
        {
            M_w[h_src[i] + num_vertices * h_dst[i]]         = h_wgt[i];
            M_edge_id[h_src[i] + num_vertices * h_dst[i]]   = h_edge_ids[i];
            M_edge_type[h_src[i] + num_vertices * h_dst[i]] = h_edge_types[i];
        }

        if(renumber_results)
        {
            for(int label_id = 0; label_id < (result_offsets_size - 1); ++label_id)
            {
                for(size_t i = h_result_offsets[label_id]; i < h_result_offsets[label_id + 1]; ++i)
                {
                    const vertex_t src
                        = h_renumber_map[h_renumber_map_offsets[label_id] + h_result_srcs[i]];
                    const vertex_t dst
                        = h_renumber_map[h_renumber_map_offsets[label_id] + h_result_dsts[i]];

                    ROCGRAPH_CLIENTS_EXPECT_EQ(M_w[src + num_vertices * dst], h_result_weight[i]);
                    ROCGRAPH_CLIENTS_EXPECT_EQ(M_edge_id[src + num_vertices * dst],
                                               h_result_edge_id[i]);
                    ROCGRAPH_CLIENTS_EXPECT_EQ(M_edge_type[src + num_vertices * dst],
                                               h_result_edge_types[i]);
                }
            }
        }
        else
        {
            for(int i = 0; i < result_size; ++i)
            {
                ROCGRAPH_CLIENTS_EXPECT_EQ(M_w[h_result_srcs[i] + num_vertices * h_result_dsts[i]],
                                           h_result_weight[i]);
                ROCGRAPH_CLIENTS_EXPECT_EQ(
                    M_edge_id[h_result_srcs[i] + num_vertices * h_result_dsts[i]],
                    h_result_edge_id[i]);
                ROCGRAPH_CLIENTS_EXPECT_EQ(
                    M_edge_type[h_result_srcs[i] + num_vertices * h_result_dsts[i]],
                    h_result_edge_types[i]);
            }
        }

        //
        // For the sampling result to make sense, all sources in hop 0 must be in the seeds,
        // all sources in hop 1 must be a result from hop 0, etc.
        //
        std::vector<vertex_t> check_v1(result_size);
        std::vector<vertex_t> check_v2(result_size);
        vertex_t*             check_sources      = (vertex_t*)check_v1.data();
        vertex_t*             check_destinations = (vertex_t*)check_v2.data();

        std::vector<size_t> degree(num_vertices);
        for(size_t i = 0; i < num_vertices; ++i)
            degree[i] = 0;

        for(size_t i = 0; i < num_edges; ++i)
        {
            degree[h_src[i]]++;
        }

        for(int label_id = 0; label_id < (result_offsets_size - 1); ++label_id)
        {
            size_t sources_size      = 0;
            size_t destinations_size = 0;

            // Fill sources with the input sources
            for(size_t i = 0; i < num_start_vertices; ++i)
            {
                if(h_start_labels[i] == h_result_labels[label_id])
                {
                    check_sources[sources_size] = h_start[i];
                    ++sources_size;
                }
            }

            if(renumber_results)
            {
                size_t num_vertex_ids
                    = 2 * (h_result_offsets[label_id + 1] - h_result_offsets[label_id]);
                std::vector<vertex_t> vertex_ids(num_vertex_ids);

                for(size_t i = 0; i < (h_result_offsets[label_id + 1] - h_result_offsets[label_id]);
                    ++i)
                {
                    vertex_ids[2 * i]     = h_result_srcs[h_result_offsets[label_id] + i];
                    vertex_ids[2 * i + 1] = h_result_dsts[h_result_offsets[label_id] + i];
                }

                qsort(vertex_ids.data(),
                      num_vertex_ids,
                      sizeof(vertex_t),
                      vertex_id_compare_function<vertex_t>);

                vertex_t current_v = 0;
                for(size_t i = 0; i < num_vertex_ids; ++i)
                {
                    if(vertex_ids[i] == current_v)
                        ++current_v;
                    else
                        ROCGRAPH_CLIENTS_EXPECT_EQ(vertex_ids[i], (current_v - 1));
                }
            }

            for(int hop = 0; hop < fan_out_size; ++hop)
            {
                if(prior_sources_behavior == rocgraph_prior_sources_behavior_carry_over)
                {
                    destinations_size = sources_size;
                    for(size_t i = 0; i < sources_size; ++i)
                    {
                        check_destinations[i] = check_sources[i];
                    }
                }

                for(size_t i = h_result_offsets[label_id]; i < h_result_offsets[label_id + 1]; ++i)
                {
                    if(h_result_hops[i] == hop)
                    {

                        bool found = false;
                        for(size_t j = 0; (!found) && (j < sources_size); ++j)
                        {
                            found = renumber_results
                                        ? (h_renumber_map[h_renumber_map_offsets[label_id]
                                                          + h_result_srcs[i]]
                                           == check_sources[j])
                                        : (h_result_srcs[i] == check_sources[j]);
                        }

                        ROCGRAPH_CLIENTS_EXPECT_TRUE(found);
                    }

                    if(prior_sources_behavior == rocgraph_prior_sources_behavior_carry_over)
                    {
                        // Make sure destination isn't already in the source list
                        bool found = false;
                        for(size_t j = 0; (!found) && (j < destinations_size); ++j)
                        {
                            found = renumber_results
                                        ? (h_renumber_map[h_renumber_map_offsets[label_id]
                                                          + h_result_dsts[i]]
                                           == check_destinations[j])
                                        : (h_result_dsts[i] == check_destinations[j]);
                        }

                        if(!found)
                        {
                            check_destinations[destinations_size]
                                = renumber_results ? h_renumber_map[h_renumber_map_offsets[label_id]
                                                                    + h_result_dsts[i]]
                                                   : h_result_dsts[i];
                            ++destinations_size;
                        }
                    }
                    else
                    {
                        check_destinations[destinations_size]
                            = renumber_results ? h_renumber_map[h_renumber_map_offsets[label_id]
                                                                + h_result_dsts[i]]
                                               : h_result_dsts[i];
                        ++destinations_size;
                    }
                }

                vertex_t* tmp      = check_sources;
                check_sources      = check_destinations;
                check_destinations = tmp;
                sources_size       = destinations_size;
                destinations_size  = 0;
            }

            if(prior_sources_behavior == rocgraph_prior_sources_behavior_exclude)
            {
                // Make sure vertex v only appears as source in the first hop after it is encountered
                for(size_t i = h_result_offsets[label_id]; i < h_result_offsets[label_id + 1]; ++i)
                {
                    for(size_t j = i + 1; j < h_result_offsets[label_id + 1]; ++j)
                    {
                        if(h_result_srcs[i] == h_result_srcs[j])
                        {
                            ROCGRAPH_CLIENTS_EXPECT_EQ(h_result_hops[i], h_result_hops[j]);
                        }
                    }
                }
            }

            if(dedupe_sources)
            {
                // Make sure vertex v only appears as source once for each edge after it appears as destination
                // Externally test this by verifying that vertex v only appears in <= hop size/degree
                for(size_t i = h_result_offsets[label_id]; i < h_result_offsets[label_id + 1]; ++i)
                {
                    if(h_result_hops[i] > 0)
                    {
                        size_t num_occurrences = 1;
                        for(size_t j = i + 1; j < h_result_offsets[label_id + 1]; ++j)
                        {
                            if((h_result_srcs[j] == h_result_srcs[i])
                               && (h_result_hops[j] == h_result_hops[i]))
                                num_occurrences++;
                        }

                        if(fan_out[h_result_hops[i]] < 0)
                        {
                            ROCGRAPH_CLIENTS_EXPECT_LE(num_occurrences, degree[h_result_srcs[i]]);
                        }
                        else
                        {
                            ROCGRAPH_CLIENTS_EXPECT_LT(num_occurrences,
                                                       size_t(fan_out[h_result_hops[i]]));
                        }
                    }
                }
            }
        }

        rocgraph_sample_result_free(result);
        // #endif

        rocgraph_sg_graph_free(graph);
        rocgraph_error_free(ret_error);
    }

    void UniformNeighborSampleClean(const Arguments& arg)
    {
        using vertex_t = int32_t;
        using edge_t   = int32_t;
        using weight_t = float;

        ROCGRAPH_CLIENTS_SKIP_TEST("Skipping UniformNeighborSampleClean because ROCGRAPH_OPS is "
                                   "not supported in this release");

        size_t num_edges        = 9;
        size_t num_vertices     = 6;
        size_t fan_out_size     = 3;
        size_t num_starts       = 2;
        size_t num_start_labels = 2;

        vertex_t src[]          = {0, 0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]          = {1, 3, 3, 4, 0, 1, 3, 5, 5};
        edge_t   edge_ids[]     = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        weight_t weight[]       = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
        int32_t  edge_types[]   = {8, 7, 6, 5, 4, 3, 2, 1, 0};
        vertex_t start[]        = {2, 3};
        int      start_labels[] = {6, 12};
        int      fan_out[]      = {-1, -1, -1};

        rocgraph_bool                   with_replacement = rocgraph_bool_false;
        rocgraph_bool                   return_hops      = rocgraph_bool_true;
        rocgraph_prior_sources_behavior prior_sources_behavior
            = rocgraph_prior_sources_behavior_default;
        rocgraph_bool dedupe_sources   = rocgraph_bool_false;
        rocgraph_bool renumber_results = rocgraph_bool_false;

        generic_uniform_neighbor_sample_test(src,
                                             dst,
                                             weight,
                                             edge_ids,
                                             edge_types,
                                             num_vertices,
                                             num_edges,
                                             start,
                                             start_labels,
                                             num_starts,
                                             num_start_labels,
                                             fan_out,
                                             fan_out_size,
                                             with_replacement,
                                             return_hops,
                                             prior_sources_behavior,
                                             dedupe_sources,
                                             renumber_results);
    }

    void UniformNeighborSampleDedupeSources(const Arguments& arg)
    {
        using vertex_t = int32_t;
        using edge_t   = int32_t;
        using weight_t = float;

        ROCGRAPH_CLIENTS_SKIP_TEST("Skipping UniformNeighborSampleDedupeSources because "
                                   "ROCGRAPH_OPS is not supported in this release");

        size_t num_edges        = 9;
        size_t num_vertices     = 6;
        size_t fan_out_size     = 3;
        size_t num_starts       = 2;
        size_t num_start_labels = 2;

        vertex_t src[]          = {0, 0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]          = {1, 3, 3, 4, 0, 1, 3, 5, 5};
        edge_t   edge_ids[]     = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        weight_t weight[]       = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
        int32_t  edge_types[]   = {8, 7, 6, 5, 4, 3, 2, 1, 0};
        vertex_t start[]        = {2, 3};
        int      start_labels[] = {6, 12};
        int      fan_out[]      = {-1, -1, -1};

        rocgraph_bool                   with_replacement = rocgraph_bool_false;
        rocgraph_bool                   return_hops      = rocgraph_bool_true;
        rocgraph_prior_sources_behavior prior_sources_behavior
            = rocgraph_prior_sources_behavior_default;
        rocgraph_bool dedupe_sources   = rocgraph_bool_true;
        rocgraph_bool renumber_results = rocgraph_bool_false;

        generic_uniform_neighbor_sample_test(src,
                                             dst,
                                             weight,
                                             edge_ids,
                                             edge_types,
                                             num_vertices,
                                             num_edges,
                                             start,
                                             start_labels,
                                             num_starts,
                                             num_start_labels,
                                             fan_out,
                                             fan_out_size,
                                             with_replacement,
                                             return_hops,
                                             prior_sources_behavior,
                                             dedupe_sources,
                                             renumber_results);
    }

    void UniformNeighborSampleUniqueSources(const Arguments& arg)
    {
        using vertex_t = int32_t;
        using edge_t   = int32_t;
        using weight_t = float;

        ROCGRAPH_CLIENTS_SKIP_TEST("Skipping UniformNeighborSampleUniqueSources because "
                                   "ROCGRAPH_OPS is not supported in this release");

        size_t num_edges        = 9;
        size_t num_vertices     = 6;
        size_t fan_out_size     = 3;
        size_t num_starts       = 2;
        size_t num_start_labels = 2;

        vertex_t src[]          = {0, 0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]          = {1, 2, 3, 4, 0, 1, 3, 5, 5};
        edge_t   edge_ids[]     = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        weight_t weight[]       = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
        int32_t  edge_types[]   = {8, 7, 6, 5, 4, 3, 2, 1, 0};
        vertex_t start[]        = {2, 3};
        int      start_labels[] = {6, 12};
        int      fan_out[]      = {-1, -1, -1};

        rocgraph_bool                   with_replacement = rocgraph_bool_false;
        rocgraph_bool                   return_hops      = rocgraph_bool_true;
        rocgraph_prior_sources_behavior prior_sources_behavior
            = rocgraph_prior_sources_behavior_exclude;
        rocgraph_bool dedupe_sources   = rocgraph_bool_false;
        rocgraph_bool renumber_results = rocgraph_bool_false;

        generic_uniform_neighbor_sample_test(src,
                                             dst,
                                             weight,
                                             edge_ids,
                                             edge_types,
                                             num_vertices,
                                             num_edges,
                                             start,
                                             start_labels,
                                             num_starts,
                                             num_start_labels,
                                             fan_out,
                                             fan_out_size,
                                             with_replacement,
                                             return_hops,
                                             prior_sources_behavior,
                                             dedupe_sources,
                                             renumber_results);
    }

    void UniformNeighborSampleCarryOverSources(const Arguments& arg)
    {
        using vertex_t = int32_t;
        using edge_t   = int32_t;
        using weight_t = float;

        ROCGRAPH_CLIENTS_SKIP_TEST("Skipping UniformNeighborSampleCarryOverSources because "
                                   "ROCGRAPH_OPS is not supported in this release");

        size_t num_edges        = 9;
        size_t num_vertices     = 6;
        size_t fan_out_size     = 3;
        size_t num_starts       = 2;
        size_t num_start_labels = 2;

        vertex_t src[]          = {0, 0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]          = {1, 2, 3, 4, 0, 1, 3, 5, 5};
        edge_t   edge_ids[]     = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        weight_t weight[]       = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
        int32_t  edge_types[]   = {8, 7, 6, 5, 4, 3, 2, 1, 0};
        vertex_t start[]        = {2, 3};
        int      start_labels[] = {6, 12};
        int      fan_out[]      = {-1, -1, -1};

        rocgraph_bool                   with_replacement = rocgraph_bool_false;
        rocgraph_bool                   return_hops      = rocgraph_bool_true;
        rocgraph_prior_sources_behavior prior_sources_behavior
            = rocgraph_prior_sources_behavior_carry_over;
        rocgraph_bool dedupe_sources   = rocgraph_bool_false;
        rocgraph_bool renumber_results = rocgraph_bool_false;

        generic_uniform_neighbor_sample_test(src,
                                             dst,
                                             weight,
                                             edge_ids,
                                             edge_types,
                                             num_vertices,
                                             num_edges,
                                             start,
                                             start_labels,
                                             num_starts,
                                             num_start_labels,
                                             fan_out,
                                             fan_out_size,
                                             with_replacement,
                                             return_hops,
                                             prior_sources_behavior,
                                             dedupe_sources,
                                             renumber_results);
    }

    void UniformNeighborSampleRenumberResults(const Arguments& arg)
    {
        using vertex_t = int32_t;
        using edge_t   = int32_t;
        using weight_t = float;

        ROCGRAPH_CLIENTS_SKIP_TEST("Skipping UniformNeighborSampleRenumberResults because "
                                   "ROCGRAPH_OPS is not supported in this release");

        size_t num_edges        = 9;
        size_t num_vertices     = 6;
        size_t fan_out_size     = 3;
        size_t num_starts       = 2;
        size_t num_start_labels = 2;

        vertex_t src[]          = {0, 0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]          = {1, 2, 3, 4, 0, 1, 3, 5, 5};
        edge_t   edge_ids[]     = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        weight_t weight[]       = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
        int32_t  edge_types[]   = {8, 7, 6, 5, 4, 3, 2, 1, 0};
        vertex_t start[]        = {2, 3};
        int      start_labels[] = {6, 12};
        int      fan_out[]      = {-1, -1, -1};

        rocgraph_bool                   with_replacement = rocgraph_bool_false;
        rocgraph_bool                   return_hops      = rocgraph_bool_true;
        rocgraph_prior_sources_behavior prior_sources_behavior
            = rocgraph_prior_sources_behavior_default;
        rocgraph_bool dedupe_sources   = rocgraph_bool_false;
        rocgraph_bool renumber_results = rocgraph_bool_true;

        generic_uniform_neighbor_sample_test(src,
                                             dst,
                                             weight,
                                             edge_ids,
                                             edge_types,
                                             num_vertices,
                                             num_edges,
                                             start,
                                             start_labels,
                                             num_starts,
                                             num_start_labels,
                                             fan_out,
                                             fan_out_size,
                                             with_replacement,
                                             return_hops,
                                             prior_sources_behavior,
                                             dedupe_sources,
                                             renumber_results);
    }

    void UniformNeighborSampleWithLabels(const Arguments& arg)
    {
        using vertex_t = int32_t;
        using edge_t   = int32_t;
        using weight_t = float;

        ROCGRAPH_CLIENTS_SKIP_TEST("Skipping UniformNeighborSampleWithLabels because ROCGRAPH_OPS "
                                   "is not supported in this release");

        rocgraph_data_type_id vertex_tid    = rocgraph_data_type_id_int32;
        rocgraph_data_type_id edge_tid      = rocgraph_data_type_id_int32;
        rocgraph_data_type_id weight_tid    = rocgraph_data_type_id_float32;
        rocgraph_data_type_id edge_id_tid   = rocgraph_data_type_id_int32;
        rocgraph_data_type_id edge_type_tid = rocgraph_data_type_id_int32;

        size_t num_edges = 8;

        // size_t fan_out_size = 1;
        size_t num_starts = 2;

        vertex_t src[]          = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]          = {1, 3, 4, 0, 1, 3, 5, 5};
        edge_t   edge_ids[]     = {0, 1, 2, 3, 4, 5, 6, 7};
        weight_t weight[]       = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
        int32_t  edge_types[]   = {7, 6, 5, 4, 3, 2, 1, 0};
        vertex_t start[]        = {2, 3};
        size_t   start_labels[] = {6, 12};
        int      fan_out[]      = {-1};

        // Create graph
        rocgraph_error_t*         ret_error = nullptr;
        rocgraph_graph_t*         graph     = nullptr;
        rocgraph_sample_result_t* result    = nullptr;

        rocgraph_bool                   with_replacement = rocgraph_bool_true;
        rocgraph_bool                   return_hops      = rocgraph_bool_true;
        rocgraph_prior_sources_behavior prior_sources_behavior
            = rocgraph_prior_sources_behavior_default;
        rocgraph_bool             dedupe_sources   = rocgraph_bool_false;
        rocgraph_bool             renumber_results = rocgraph_bool_false;
        rocgraph_compression_type compression      = rocgraph_compression_type_coo;
        rocgraph_bool             compress_per_hop = rocgraph_bool_false;

        rocgraph_handle_t* p_handle = nullptr;
        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        rocgraph_clients_create_sg_test_graph(p_handle,
                                              vertex_tid,
                                              edge_tid,
                                              src,
                                              dst,
                                              weight_tid,
                                              weight,
                                              edge_type_tid,
                                              edge_types,
                                              edge_id_tid,
                                              edge_ids,
                                              num_edges,
                                              rocgraph_bool_false,
                                              rocgraph_bool_true,
                                              rocgraph_bool_false,
                                              rocgraph_bool_false,
                                              &graph,
                                              &ret_error);

        rocgraph_type_erased_device_array_t*      d_start             = nullptr;
        rocgraph_type_erased_device_array_view_t* d_start_view        = nullptr;
        rocgraph_type_erased_device_array_t*      d_start_labels      = nullptr;
        rocgraph_type_erased_device_array_view_t* d_start_labels_view = nullptr;
        rocgraph_type_erased_host_array_view_t*   h_fan_out_view      = nullptr;

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_create(
                p_handle, num_starts, rocgraph_data_type_id_int32, &d_start, &ret_error),
            ret_error);

        d_start_view = rocgraph_type_erased_device_array_view(d_start);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_from_host(
                p_handle, d_start_view, (rocgraph_byte_t*)start, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_create(
                p_handle, num_starts, rocgraph_data_type_id_int32, &d_start_labels, &ret_error),
            ret_error);

        d_start_labels_view = rocgraph_type_erased_device_array_view(d_start_labels);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_from_host(
                p_handle, d_start_labels_view, (rocgraph_byte_t*)start_labels, &ret_error),
            ret_error);

        h_fan_out_view
            = rocgraph_type_erased_host_array_view_create(fan_out, 1, rocgraph_data_type_id_int32);

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

        auto ret_code = rocgraph_uniform_neighbor_sample(p_handle,
                                                         graph,
                                                         d_start_view,
                                                         d_start_labels_view,
                                                         nullptr,
                                                         nullptr,
                                                         nullptr,
                                                         h_fan_out_view,
                                                         rng_state,
                                                         sampling_options,
                                                         rocgraph_bool_false,
                                                         &result,
                                                         &ret_error);

#ifdef NO_ROCGRAPH_OPS
        ROCGRAPH_CLIENTS_EXPECT_NE(ret_code, rocgraph_status_success);
#else
        CHECK_ROCGRAPH_SUCCESS_ERROR(ret_code, ret_error);

        size_t                                    num_vertices = 6;
        rocgraph_type_erased_device_array_view_t* result_srcs;
        rocgraph_type_erased_device_array_view_t* result_dsts;
        rocgraph_type_erased_device_array_view_t* result_edge_id;
        rocgraph_type_erased_device_array_view_t* result_weights;
        rocgraph_type_erased_device_array_view_t* result_edge_types;
        rocgraph_type_erased_device_array_view_t* result_hops;
        rocgraph_type_erased_device_array_view_t* result_offsets;

        result_srcs       = rocgraph_sample_result_get_sources(result);
        result_dsts       = rocgraph_sample_result_get_destinations(result);
        result_edge_id    = rocgraph_sample_result_get_edge_id(result);
        result_weights    = rocgraph_sample_result_get_edge_weight(result);
        result_edge_types = rocgraph_sample_result_get_edge_type(result);
        result_hops       = rocgraph_sample_result_get_hop(result);
        result_offsets    = rocgraph_sample_result_get_offsets(result);

        size_t result_size         = rocgraph_type_erased_device_array_view_size(result_srcs);
        size_t result_offsets_size = rocgraph_type_erased_device_array_view_size(result_offsets);

        std::vector<vertex_t> h_srcs(result_size);
        std::vector<vertex_t> h_dsts(result_size);
        std::vector<edge_t>   h_edge_id(result_size);
        std::vector<weight_t> h_weight(result_size);
        std::vector<int32_t>  h_edge_types(result_size);
        // std::vector<int32_t>  h_hops(result_size);
        std::vector<size_t> h_result_offsets(result_offsets_size);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_srcs.data(), result_srcs, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_dsts.data(), result_dsts, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_edge_id.data(), result_edge_id, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_weight.data(), result_weights, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_edge_types.data(), result_edge_types, &ret_error),
            ret_error);

        ROCGRAPH_CLIENTS_EXPECT_EQ(result_hops, nullptr);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_offsets.data(), result_offsets, &ret_error),
            ret_error);

        //  NOTE:  The C++ tester does a more thorough validation.  For our purposes
        //  here we will do a simpler validation, merely checking that all edges
        //  are actually part of the graph
        std::vector<weight_t> M_w(num_vertices * num_vertices);
        std::vector<edge_t>   M_edge_id(num_vertices * num_vertices);
        std::vector<int32_t>  M_edge_type(num_vertices * num_vertices);

        for(int i = 0; i < num_vertices; ++i)
            for(int j = 0; j < num_vertices; ++j)
            {
                M_w[i + num_vertices * j]         = 0.0;
                M_edge_id[i + num_vertices * j]   = -1;
                M_edge_type[i + num_vertices * j] = -1;
            }

        for(int i = 0; i < num_edges; ++i)
        {
            M_w[src[i] + num_vertices * dst[i]]         = weight[i];
            M_edge_id[src[i] + num_vertices * dst[i]]   = edge_ids[i];
            M_edge_type[src[i] + num_vertices * dst[i]] = edge_types[i];
        }

        for(int i = 0; i < result_size; ++i)
        {
            ROCGRAPH_CLIENTS_EXPECT_EQ(M_w[h_srcs[i] + num_vertices * h_dsts[i]], h_weight[i]);
            ROCGRAPH_CLIENTS_EXPECT_EQ(M_edge_id[h_srcs[i] + num_vertices * h_dsts[i]],
                                       h_edge_id[i]);
            ROCGRAPH_CLIENTS_EXPECT_EQ(M_edge_type[h_srcs[i] + num_vertices * h_dsts[i]],
                                       h_edge_types[i]);
        }

        rocgraph_sample_result_free(result);
        rocgraph_sampling_options_free(sampling_options);
#endif

        rocgraph_sg_graph_free(graph);
        rocgraph_error_free(ret_error);
    }

/* This method is not used anywhere */
#if 0

    void create_test_graph_with_edge_ids(const rocgraph_handle_t* p_handle,
                                         vertex_t*                         h_src,
                                         vertex_t*                         h_dst,
                                         edge_t*                           h_ids,
                                         size_t                            num_edges,
                                         rocgraph_bool                   store_transposed,
                                         rocgraph_bool                   renumber,
                                         rocgraph_bool                   is_symmetric,
                                         rocgraph_graph_t**                p_graph,
                                         rocgraph_error_t**                ret_error)
    {
        rocgraph_graph_properties_t properties;

        properties.is_symmetric  = is_symmetric;
        properties.is_multigraph = rocgraph_bool_false;

        rocgraph_data_type_id vertex_tid = rocgraph_data_type_id_int32;
        rocgraph_data_type_id edge_tid   = rocgraph_data_type_id_int32;
        rocgraph_data_type_id weight_tid = rocgraph_data_type_id_float32;

        rocgraph_type_erased_device_array_t*      src;
        rocgraph_type_erased_device_array_t*      dst;
        rocgraph_type_erased_device_array_t*      ids;
        rocgraph_type_erased_device_array_view_t* src_view;
        rocgraph_type_erased_device_array_view_t* dst_view;
        rocgraph_type_erased_device_array_view_t* ids_view;
        rocgraph_type_erased_device_array_view_t* wgt_view;

        rocgraph_handle_t*               p_handle        = nullptr;
        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle,nullptr));

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
									     p_handle, num_edges, vertex_tid, &src, ret_error), ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
            p_handle, num_edges, vertex_tid, &dst, ret_error), ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
            p_handle, num_edges, edge_tid, &ids, ret_error), ret_error);

        src_view = rocgraph_type_erased_device_array_view(src);
        dst_view = rocgraph_type_erased_device_array_view(dst);
        ids_view = rocgraph_type_erased_device_array_view(ids);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
            p_handle, src_view, (rocgraph_byte_t*)h_src, ret_error), ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
            p_handle, dst_view, (rocgraph_byte_t*)h_dst, ret_error), ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
            p_handle, ids_view, (rocgraph_byte_t*)h_ids, ret_error), ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_as_type(ids, weight_tid, &wgt_view, ret_error), ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_sg_graph_create(p_handle,
                                            &properties,
                                            src_view,
                                            dst_view,
                                            wgt_view,
                                            nullptr,
                                            nullptr,
                                            store_transposed,
                                            renumber,
                                            rocgraph_bool_false,
                                            p_graph,
							     ret_error),
				    ret_error);

        rocgraph_type_erased_device_array_view_free(wgt_view);
        rocgraph_type_erased_device_array_view_free(ids_view);
        rocgraph_type_erased_device_array_view_free(dst_view);
        rocgraph_type_erased_device_array_view_free(src_view);
        rocgraph_type_erased_device_array_free(ids);
        rocgraph_type_erased_device_array_free(dst);
        rocgraph_type_erased_device_array_free(src);
    }
#endif
} // namespace

template <typename T>
void testing_rocgraph_uniform_neighbor_sample_bad_arg(const Arguments& arg)
{
#ifdef TO_FILL
    const rocgraph_handle_t*                        handle{};
    rocgraph_graph_t*                               graph{};
    const rocgraph_type_erased_device_array_view_t* start_vertices{};
    const rocgraph_type_erased_device_array_view_t* start_vertex_labels{};
    const rocgraph_type_erased_device_array_view_t* label_list{};
    const rocgraph_type_erased_device_array_view_t* label_to_comm_rank{};
    const rocgraph_type_erased_device_array_view_t* label_offsets{};
    const rocgraph_type_erased_host_array_view_t*   fan_out{};
    rocgraph_rng_state_t*                           rng_state{};
    const rocgraph_sampling_options_t*              options{};
    rocgraph_bool                                   do_expensive_check{};
    rocgraph_sample_result_t**                      result{};
    rocgraph_error_t**                              error{};
    auto                                            ret = rocgraph_uniform_neighbor_sample(handle,
                                                graph,
                                                start_vertices,
                                                start_vertex_labels,
                                                label_list,
                                                label_to_comm_rank,
                                                label_offsets,
                                                fan_out,
                                                rng_state,
                                                options,
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
void testing_rocgraph_uniform_neighbor_sample(const Arguments& arg)
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

#define INSTANTIATE(TYPE)                                                                       \
    template void testing_rocgraph_uniform_neighbor_sample_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_rocgraph_uniform_neighbor_sample<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_uniform_neighbor_sample_extra(const Arguments& arg)
{
    testing_dispatch_extra(arg,
                           "UniformNeighborSampleClean",
                           UniformNeighborSampleClean,
                           "UniformNeighborSampleDedupeSources",
                           UniformNeighborSampleDedupeSources,
                           "UniformNeighborSampleUniqueSources",
                           UniformNeighborSampleUniqueSources,
                           "UniformNeighborSampleCarryOverSources",
                           UniformNeighborSampleCarryOverSources,
                           "UniformNeighborSampleRenumberResults",
                           UniformNeighborSampleRenumberResults,
                           "UniformNeighborSampleWithLabels",
                           UniformNeighborSampleWithLabels);
}
