// Copyright (C) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "testing_rocgraph_k_truss_subgraph.hpp"
#include "rocgraph/rocgraph.h"
#include "rocgraph_clients_create_test_graph.hpp"
#include "rocgraph_clients_skip_test.hpp"
#include "rocgraph_test.hpp"
#include "testing.hpp"
namespace
{

    template <typename weight_t, typename vertex_t>
    static void generic_k_truss_test(vertex_t*     h_src,
                                     vertex_t*     h_dst,
                                     weight_t*     h_wgt,
                                     vertex_t*     h_expected_src,
                                     vertex_t*     h_expected_dst,
                                     weight_t*     h_expected_wgt,
                                     size_t*       h_expected_offsets,
                                     size_t        num_vertices,
                                     size_t        num_edges,
                                     size_t        k,
                                     size_t        num_expected_offsets,
                                     size_t        num_expected_edges,
                                     rocgraph_bool store_transposed)
    {

        rocgraph_error_t* ret_error;

        rocgraph_data_type_id vertex_tid    = rocgraph_data_type_id_int32;
        rocgraph_data_type_id edge_tid      = rocgraph_data_type_id_int32;
        rocgraph_data_type_id weight_tid    = rocgraph_data_type_id_float32;
        rocgraph_data_type_id edge_id_tid   = rocgraph_data_type_id_int32;
        rocgraph_data_type_id edge_type_tid = rocgraph_data_type_id_int32;

        rocgraph_handle_t* p_handle = nullptr;
        rocgraph_graph_t*  graph    = nullptr;
        // rocgraph_type_erased_device_array_t* seeds           = nullptr;
        // rocgraph_type_erased_device_array_view_t* seeds_view = nullptr;
        rocgraph_induced_subgraph_result_t* result = nullptr;

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

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_k_truss_subgraph(p_handle, graph, k, rocgraph_bool_false, &result, &ret_error),
            ret_error);

        rocgraph_type_erased_device_array_view_t* src;
        rocgraph_type_erased_device_array_view_t* dst;
        rocgraph_type_erased_device_array_view_t* wgt;
        rocgraph_type_erased_device_array_view_t* offsets;

        src     = rocgraph_induced_subgraph_get_sources(result);
        dst     = rocgraph_induced_subgraph_get_destinations(result);
        wgt     = rocgraph_induced_subgraph_get_edge_weights(result);
        offsets = rocgraph_induced_subgraph_get_subgraph_offsets(result);

        size_t num_result_edges   = rocgraph_type_erased_device_array_view_size(src);
        size_t num_result_offsets = rocgraph_type_erased_device_array_view_size(offsets);

        std::vector<vertex_t> h_result_src(num_result_edges);
        std::vector<vertex_t> h_result_dst(num_result_edges);
        std::vector<weight_t> h_result_wgt(num_result_edges);
        std::vector<size_t>   h_result_offsets(num_result_offsets);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_src.data(), src, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_dst.data(), dst, &ret_error),
            ret_error);

        if(wgt != nullptr)
        {
            CHECK_ROCGRAPH_SUCCESS_ERROR(
                rocgraph_type_erased_device_array_view_copy_to_host(
                    p_handle, (rocgraph_byte_t*)h_result_wgt.data(), wgt, &ret_error),
                ret_error);
        }

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_result_offsets.data(), offsets, &ret_error),
            ret_error);

        ROCGRAPH_CLIENTS_EXPECT_EQ(num_result_edges, num_expected_edges);
        ROCGRAPH_CLIENTS_EXPECT_SEGMENT_EQ(
            num_expected_offsets, h_expected_offsets, h_result_offsets.data());

        for(size_t i = 0; i < num_expected_edges; ++i)
        {
            rocgraph_bool found = rocgraph_bool_false;
            for(size_t j = 0; (j < num_expected_edges) && !found; ++j)
            {
                if((h_expected_src[i] == h_result_src[j]) && (h_expected_dst[i] == h_result_dst[j]))
                {
                    ROCGRAPH_CLIENTS_EXPECT_NEAR_TOLERANCE(
                        h_expected_wgt[i], h_result_wgt[j], 0.001);
                }
            }
        }

        rocgraph_type_erased_device_array_view_free(src);
        rocgraph_type_erased_device_array_view_free(dst);
        rocgraph_type_erased_device_array_view_free(wgt);
        rocgraph_type_erased_device_array_view_free(offsets);
        rocgraph_induced_subgraph_result_free(result);

        rocgraph_sg_graph_free(graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);

        //
    };

    void KTruss(const Arguments& arg)
    {
        using vertex_t = int32_t;
        using weight_t = float;
        ROCGRAPH_CLIENTS_SKIP_TEST(
            "Skipping KTruss because of because it is not supported in this release");
        size_t num_edges    = 16;
        size_t num_vertices = 6;
        size_t k            = 3;

        vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
        weight_t h_wgt[] = {0.1f,
                            2.1f,
                            1.1f,
                            5.1f,
                            3.1f,
                            4.1f,
                            7.2f,
                            3.2f,
                            0.1f,
                            2.1f,
                            1.1f,
                            5.1f,
                            3.1f,
                            4.1f,
                            7.2f,
                            3.2f};

        vertex_t h_result_src[]       = {1, 2, 2, 3, 3, 0, 0, 1, 1, 2};
        vertex_t h_result_dst[]       = {0, 0, 1, 1, 2, 1, 2, 2, 3, 3};
        weight_t h_result_wgt[]       = {0.1, 5.1, 3.1, 2.1, 4.1, 0.1, 5.1, 3.1, 2.1, 4.1};
        size_t   h_result_offsets[]   = {0, 10};
        size_t   num_expected_edges   = 10;
        size_t   num_expected_offsets = 2;

        generic_k_truss_test<weight_t, vertex_t>(h_src,
                                                 h_dst,
                                                 h_wgt,
                                                 h_result_src,
                                                 h_result_dst,
                                                 h_result_wgt,
                                                 h_result_offsets,
                                                 num_vertices,
                                                 num_edges,
                                                 k,
                                                 num_expected_offsets,
                                                 num_expected_edges,
                                                 rocgraph_bool_false);
    };

    void KTrussNoWeights(const Arguments& arg)
    {
        using vertex_t = int32_t;
        using weight_t = float;
        ROCGRAPH_CLIENTS_SKIP_TEST(
            "Skipping KTrussNoWeights because of because it is not supported in this release");
        size_t num_edges    = 16;
        size_t num_vertices = 6;
        size_t k            = 3;

        vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};

        vertex_t h_result_src[]       = {0, 0, 2, 2, 3, 1, 2, 1, 3, 1};
        vertex_t h_result_dst[]       = {1, 2, 1, 3, 1, 0, 0, 2, 2, 3};
        size_t   h_result_offsets[]   = {0, 10};
        size_t   num_expected_edges   = 10;
        size_t   num_expected_offsets = 2;

        generic_k_truss_test<weight_t, vertex_t>(h_src,
                                                 h_dst,
                                                 nullptr,
                                                 h_result_src,
                                                 h_result_dst,
                                                 nullptr,
                                                 h_result_offsets,
                                                 num_vertices,
                                                 num_edges,
                                                 k,
                                                 num_expected_offsets,
                                                 num_expected_edges,
                                                 rocgraph_bool_false);
    };
};

template <typename T>
void testing_rocgraph_k_truss_subgraph_bad_arg(const Arguments& arg)
{
#ifdef TO_FILL
    const rocgraph_handle_t*             handle{};
    rocgraph_graph_t*                    graph{};
    size_t                               k{};
    rocgraph_bool                        do_expensive_check{};
    rocgraph_induced_subgraph_result_t** result{};
    rocgraph_error_t**                   error{};
    auto ret = rocgraph_k_truss_subgraph(handle, graph, k, do_expensive_check, result, error);
    if(ret != rocgraph_status_success)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_internal_error);
    }
    CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
#endif
}

template <typename T>
void testing_rocgraph_k_truss_subgraph(const Arguments& arg)
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

#define INSTANTIATE(TYPE)                                                                \
    template void testing_rocgraph_k_truss_subgraph_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_rocgraph_k_truss_subgraph<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_k_truss_subgraph_extra(const Arguments& arg)
{
    testing_dispatch_extra(arg, "KTruss", KTruss, "KTrussNoWeights", KTrussNoWeights);
}
