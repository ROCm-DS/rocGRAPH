/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "testing_rocgraph_balanced_cut_clustering.hpp"
#include "rocgraph/rocgraph.h"
#include "rocgraph_clients_create_test_graph.hpp"
#include "rocgraph_clients_skip_test.hpp"
#include "rocgraph_test.hpp"
#include "testing.hpp"

namespace
{
    template <typename weight_t, typename edge_t, typename vertex_t>
    void generic_spectral_test(vertex_t*     h_src,
                               vertex_t*     h_dst,
                               weight_t*     h_wgt,
                               vertex_t*     h_result,
                               weight_t      expected_modularity,
                               weight_t      expected_edge_cut,
                               weight_t      expected_ratio_cut,
                               size_t        num_vertices,
                               size_t        num_edges,
                               size_t        num_clusters,
                               size_t        num_eigenvectors,
                               double        evs_tolerance,
                               int           evs_max_iterations,
                               double        k_means_tolerance,
                               int           k_means_max_iterations,
                               rocgraph_bool store_transposed)
    {

        rocgraph_error_t* ret_error;

        rocgraph_handle_t*            p_handle = nullptr;
        rocgraph_graph_t*             graph    = nullptr;
        rocgraph_clustering_result_t* result   = nullptr;

        rocgraph_data_type_id vertex_tid    = rocgraph_data_type_id_int32;
        rocgraph_data_type_id edge_tid      = rocgraph_data_type_id_int32;
        rocgraph_data_type_id weight_tid    = rocgraph_data_type_id_float32;
        rocgraph_data_type_id edge_id_tid   = rocgraph_data_type_id_int32;
        rocgraph_data_type_id edge_type_tid = rocgraph_data_type_id_int32;

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
            rocgraph_spectral_modularity_maximization(p_handle,
                                                      graph,
                                                      num_clusters,
                                                      num_eigenvectors,
                                                      evs_tolerance,
                                                      evs_max_iterations,
                                                      k_means_tolerance,
                                                      k_means_max_iterations,
                                                      rocgraph_bool_false,
                                                      &result,
                                                      &ret_error),
            ret_error);

        rocgraph_type_erased_device_array_view_t* vertices;
        rocgraph_type_erased_device_array_view_t* clusters;
        double                                    modularity{0.0};
        double                                    edge_cut{0.0};
        double                                    ratio_cut{0.0};

        vertices = rocgraph_clustering_result_get_vertices(result);
        clusters = rocgraph_clustering_result_get_clusters(result);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_analyze_clustering_modularity(
                p_handle, graph, num_clusters, vertices, clusters, &modularity, &ret_error),
            ret_error);

        std::vector<vertex_t> h_vertices(num_vertices);
        std::vector<edge_t>   h_clusters(num_vertices);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_vertices.data(), vertices, &ret_error),
            ret_error);
        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_clusters.data(), clusters, &ret_error),
            ret_error);

        ROCGRAPH_CLIENTS_EXPECT_INDIRECT_ARRAY_EQ(num_vertices,
                                                  h_result,
                                                  1,
                                                  h_vertices.data(),
                                                  h_clusters.data(),
                                                  1,
                                                  (const vertex_t*)nullptr);

        const double double_expected_modularity = expected_modularity;
        const double double_expected_edge_cut   = expected_edge_cut;
        const double double_expected_ratio_cut  = expected_ratio_cut;

        ROCGRAPH_CLIENTS_EXPECT_NEAR_TOLERANCE(modularity, double_expected_modularity, 0.001);
        ROCGRAPH_CLIENTS_EXPECT_NEAR_TOLERANCE(edge_cut, double_expected_edge_cut, 0.001);
        ROCGRAPH_CLIENTS_EXPECT_NEAR_TOLERANCE(ratio_cut, double_expected_ratio_cut, 0.001);

        rocgraph_clustering_result_free(result);
        rocgraph_sg_graph_free(graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    template <typename weight_t, typename edge_t, typename vertex_t>
    void generic_balanced_cut_test(vertex_t*     h_src,
                                   vertex_t*     h_dst,
                                   weight_t*     h_wgt,
                                   vertex_t*     h_result,
                                   weight_t      expected_modularity,
                                   weight_t      expected_edge_cut,
                                   weight_t      expected_ratio_cut,
                                   size_t        num_vertices,
                                   size_t        num_edges,
                                   size_t        num_clusters,
                                   size_t        num_eigenvectors,
                                   double        evs_tolerance,
                                   int           evs_max_iterations,
                                   double        k_means_tolerance,
                                   int           k_means_max_iterations,
                                   rocgraph_bool store_transposed)
    {

        rocgraph_error_t* ret_error;

        rocgraph_data_type_id vertex_tid    = rocgraph_data_type_id_int32;
        rocgraph_data_type_id edge_tid      = rocgraph_data_type_id_int32;
        rocgraph_data_type_id weight_tid    = rocgraph_data_type_id_float32;
        rocgraph_data_type_id edge_id_tid   = rocgraph_data_type_id_int32;
        rocgraph_data_type_id edge_type_tid = rocgraph_data_type_id_int32;

        rocgraph_handle_t*            p_handle = nullptr;
        rocgraph_graph_t*             graph    = nullptr;
        rocgraph_clustering_result_t* result   = nullptr;

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

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_balanced_cut_clustering(p_handle,
                                                                      graph,
                                                                      num_clusters,
                                                                      num_eigenvectors,
                                                                      evs_tolerance,
                                                                      evs_max_iterations,
                                                                      k_means_tolerance,
                                                                      k_means_max_iterations,
                                                                      rocgraph_bool_false,
                                                                      &result,
                                                                      &ret_error),
                                     ret_error);

        rocgraph_type_erased_device_array_view_t* vertices;
        rocgraph_type_erased_device_array_view_t* clusters;
        double                                    modularity;
        double                                    edge_cut;
        double                                    ratio_cut;

        vertices = rocgraph_clustering_result_get_vertices(result);
        clusters = rocgraph_clustering_result_get_clusters(result);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_analyze_clustering_modularity(
                p_handle, graph, num_clusters, vertices, clusters, &modularity, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_analyze_clustering_edge_cut(
                p_handle, graph, num_clusters, vertices, clusters, &edge_cut, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_analyze_clustering_ratio_cut(
                p_handle, graph, num_clusters, vertices, clusters, &ratio_cut, &ret_error),
            ret_error);

        std::vector<vertex_t> h_vertices(num_vertices);
        std::vector<edge_t>   h_clusters(num_vertices);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_vertices.data(), vertices, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_clusters.data(), clusters, &ret_error),
            ret_error);

        ROCGRAPH_CLIENTS_EXPECT_INDIRECT_ARRAY_EQ(num_vertices,
                                                  h_result,
                                                  1,
                                                  h_vertices.data(),
                                                  h_clusters.data(),
                                                  1,
                                                  (const vertex_t*)nullptr);

        const double double_expected_modularity = expected_modularity;
        const double double_expected_edge_cut   = expected_edge_cut;
        const double double_expected_ratio_cut  = expected_ratio_cut;

        ROCGRAPH_CLIENTS_EXPECT_NEAR_TOLERANCE(modularity, double_expected_modularity, 0.001);
        ROCGRAPH_CLIENTS_EXPECT_NEAR_TOLERANCE(edge_cut, double_expected_edge_cut, 0.001);
        ROCGRAPH_CLIENTS_EXPECT_NEAR_TOLERANCE(ratio_cut, double_expected_ratio_cut, 0.001);

        rocgraph_clustering_result_free(result);
        rocgraph_sg_graph_free(graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    void LegacySpectral(const Arguments& arg)
    {
        using vertex_t = int32_t;
        using edge_t   = int32_t;
        using weight_t = float;
        ROCGRAPH_CLIENTS_SKIP_TEST(
            "Skipping LegacySpectral because it is not supported in this release");
        size_t num_clusters           = 2;
        size_t num_eigenvectors       = 2;
        size_t num_edges              = 14;
        size_t num_vertices           = 6;
        double evs_tolerance          = 0.001;
        int    evs_max_iterations     = 100;
        double k_means_tolerance      = 0.001;
        int    k_means_max_iterations = 100;

        vertex_t h_src[] = {0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5};
        vertex_t h_dst[] = {1, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 5, 3, 4};
        weight_t h_wgt[]
            = {0.1f, 0.2f, 0.1f, 1.2f, 0.2f, 1.2f, 2.3f, 2.3f, 3.4f, 3.5f, 3.4f, 4.5f, 3.5f, 4.5f};
        vertex_t h_result[]          = {0, 0, 0, 1, 1, 1};
        weight_t expected_modularity = 0.136578;
        weight_t expected_edge_cut   = 0;
        weight_t expected_ratio_cut  = 0;

        // spectral clustering wants store_transposed = rocgraph_bool_false
        generic_spectral_test<weight_t, edge_t, vertex_t>(h_src,
                                                          h_dst,
                                                          h_wgt,
                                                          h_result,
                                                          expected_modularity,
                                                          expected_edge_cut,
                                                          expected_ratio_cut,
                                                          num_vertices,
                                                          num_edges,
                                                          num_clusters,
                                                          num_eigenvectors,
                                                          evs_tolerance,
                                                          evs_max_iterations,
                                                          k_means_tolerance,
                                                          k_means_max_iterations,
                                                          rocgraph_bool_false);
    }

    void LegacyBalancedCutUnequalWeight(const Arguments& arg)
    {
        using vertex_t = int32_t;
        using edge_t   = int32_t;
        using weight_t = float;
        ROCGRAPH_CLIENTS_SKIP_TEST(
            "Skipping LegacyBalancedCutUnequalWeight because it is not supported in this release");
        size_t num_clusters           = 2;
        size_t num_eigenvectors       = 2;
        size_t num_edges              = 14;
        size_t num_vertices           = 6;
        double evs_tolerance          = 0.001;
        int    evs_max_iterations     = 100;
        double k_means_tolerance      = 0.001;
        int    k_means_max_iterations = 100;

        vertex_t h_src[] = {0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5};
        vertex_t h_dst[] = {1, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 5, 3, 4};
        weight_t h_wgt[]
            = {0.1f, 0.2f, 0.1f, 1.2f, 0.2f, 1.2f, 2.3f, 2.3f, 3.4f, 3.5f, 3.4f, 4.5f, 3.5f, 4.5f};
        vertex_t h_result[]          = {0, 0, 1, 0, 0, 0};
        weight_t expected_modularity = -0.02963;
        weight_t expected_edge_cut   = 3.7;
        weight_t expected_ratio_cut  = 4.44;

        // balanced cut clustering wants store_transposed = rocgraph_bool_false
        generic_balanced_cut_test<weight_t, edge_t, vertex_t>(h_src,
                                                              h_dst,
                                                              h_wgt,
                                                              h_result,
                                                              expected_modularity,
                                                              expected_edge_cut,
                                                              expected_ratio_cut,
                                                              num_vertices,
                                                              num_edges,
                                                              num_clusters,
                                                              num_eigenvectors,
                                                              evs_tolerance,
                                                              evs_max_iterations,
                                                              k_means_tolerance,
                                                              k_means_max_iterations,
                                                              rocgraph_bool_false);
    }

    void LegacyBalancedCutEqualWeight(const Arguments& arg)
    {
        using vertex_t = int32_t;
        using edge_t   = int32_t;
        using weight_t = float;
        ROCGRAPH_CLIENTS_SKIP_TEST(
            "Skipping LegacyBalancedCutEqualWeight because it is not supported in this release");
        size_t num_clusters           = 2;
        size_t num_eigenvectors       = 2;
        size_t num_edges              = 14;
        size_t num_vertices           = 6;
        double evs_tolerance          = 0.001;
        int    evs_max_iterations     = 100;
        double k_means_tolerance      = 0.001;
        int    k_means_max_iterations = 100;

        vertex_t h_src[]             = {0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5};
        vertex_t h_dst[]             = {1, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 5, 3, 4};
        weight_t h_wgt[]             = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        vertex_t h_result[]          = {1, 1, 1, 0, 0, 0};
        weight_t expected_modularity = 0.357143;
        weight_t expected_edge_cut   = 1;
        weight_t expected_ratio_cut  = 0.666667;

        // balanced cut clustering wants store_transposed = rocgraph_bool_false
        generic_balanced_cut_test<weight_t, edge_t, vertex_t>(h_src,
                                                              h_dst,
                                                              h_wgt,
                                                              h_result,
                                                              expected_modularity,
                                                              expected_edge_cut,
                                                              expected_ratio_cut,
                                                              num_vertices,
                                                              num_edges,
                                                              num_clusters,
                                                              num_eigenvectors,
                                                              evs_tolerance,
                                                              evs_max_iterations,
                                                              k_means_tolerance,
                                                              k_means_max_iterations,
                                                              rocgraph_bool_false);
    }

    void LegacyBalancedCutNoWeight(const Arguments& arg)
    {
        using vertex_t = int32_t;
        using edge_t   = int32_t;
        using weight_t = float;
        ROCGRAPH_CLIENTS_SKIP_TEST(
            "Skipping LegacyBalancedCutNoWeight because it is not supported in this release");
        size_t num_clusters           = 2;
        size_t num_eigenvectors       = 2;
        size_t num_edges              = 14;
        size_t num_vertices           = 6;
        double evs_tolerance          = 0.001;
        int    evs_max_iterations     = 100;
        double k_means_tolerance      = 0.001;
        int    k_means_max_iterations = 100;

        vertex_t h_src[]             = {0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5};
        vertex_t h_dst[]             = {1, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 5, 3, 4};
        vertex_t h_result[]          = {1, 1, 1, 0, 0, 0};
        weight_t expected_modularity = 0.357143;
        weight_t expected_edge_cut   = 1;
        weight_t expected_ratio_cut  = 0.666667;

        // balanced cut clustering wants store_transposed = rocgraph_bool_false
        generic_balanced_cut_test<weight_t, edge_t, vertex_t>(h_src,
                                                              h_dst,
                                                              nullptr,
                                                              h_result,
                                                              expected_modularity,
                                                              expected_edge_cut,
                                                              expected_ratio_cut,
                                                              num_vertices,
                                                              num_edges,
                                                              num_clusters,
                                                              num_eigenvectors,
                                                              evs_tolerance,
                                                              evs_max_iterations,
                                                              k_means_tolerance,
                                                              k_means_max_iterations,
                                                              rocgraph_bool_false);
    }

} // namespace

template <typename T>
void testing_rocgraph_balanced_cut_clustering_bad_arg(const Arguments& arg)
{
#ifdef TO_FILL
    const rocgraph_handle_t*       handle{};
    rocgraph_graph_t*              graph{};
    size_t                         n_clusters{};
    size_t                         n_eigenvectors{};
    double                         evs_tolerance{};
    int                            evs_max_iterations{};
    double                         k_means_tolerance{};
    int                            k_means_max_iterations{};
    rocgraph_bool                  do_expensive_check{};
    rocgraph_clustering_result_t** result{};
    rocgraph_error_t**             error{};
    auto                           ret = rocgraph_balanced_cut_clustering(handle,
                                                graph,
                                                n_clusters,
                                                n_eigenvectors,
                                                evs_tolerance,
                                                evs_max_iterations,
                                                k_means_tolerance,
                                                k_means_max_iterations,
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
void testing_rocgraph_balanced_cut_clustering(const Arguments& arg)
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
    template void testing_rocgraph_balanced_cut_clustering_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_rocgraph_balanced_cut_clustering<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_balanced_cut_clustering_extra(const Arguments& arg)
{
    testing_dispatch_extra(arg,
                           "LegacySpectral",
                           LegacySpectral,
                           "LegacyBalancedCutUnequalWeight",
                           LegacyBalancedCutUnequalWeight,
                           "LegacyBalancedCutEqualWeight",
                           LegacyBalancedCutEqualWeight,
                           "LegacyBalancedCutNoWeight",
                           LegacyBalancedCutNoWeight);
}
