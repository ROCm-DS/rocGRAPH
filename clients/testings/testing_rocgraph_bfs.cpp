/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "testing_rocgraph_bfs.hpp"
#include "rocgraph/rocgraph.h"
#include "testing.hpp"

template <typename weight_t, typename vertex_t>
static void generic_bfs_test(vertex_t*       h_src,
                             vertex_t*       h_dst,
                             weight_t*       h_wgt,
                             vertex_t*       h_seeds,
                             const vertex_t* expected_distances,
                             const vertex_t* expected_predecessors,
                             size_t          num_vertices,
                             size_t          num_edges,
                             size_t          num_seeds,
                             size_t          depth_limit,
                             rocgraph_bool   store_transposed)
{

    rocgraph_error_t*                         ret_error     = nullptr;
    rocgraph_handle_t*                        p_handle      = nullptr;
    rocgraph_graph_t*                         p_graph       = nullptr;
    rocgraph_paths_result_t*                  p_result      = nullptr;
    rocgraph_type_erased_device_array_t*      p_sources     = nullptr;
    rocgraph_type_erased_device_array_view_t* p_source_view = nullptr;

    CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));
    CHECK_ROCGRAPH_SUCCESS((p_handle != nullptr) ? rocgraph_status_success
                                                 : rocgraph_status_invalid_handle);

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

    CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_bfs(p_handle,
                                              p_graph,
                                              p_source_view,
                                              rocgraph_bool_false,
                                              depth_limit,
                                              rocgraph_bool_true,
                                              rocgraph_bool_false,
                                              &p_result,
                                              &ret_error),
                                 ret_error);

    rocgraph_type_erased_device_array_view_t* vertices;
    rocgraph_type_erased_device_array_view_t* distances;
    rocgraph_type_erased_device_array_view_t* predecessors;

    vertices     = rocgraph_paths_result_get_vertices(p_result);
    distances    = rocgraph_paths_result_get_distances(p_result);
    predecessors = rocgraph_paths_result_get_predecessors(p_result);

    std::vector<vertex_t> h_vertices(num_vertices);
    std::vector<vertex_t> h_distances(num_vertices);
    std::vector<vertex_t> h_predecessors(num_vertices);

    CHECK_ROCGRAPH_SUCCESS_ERROR(
        rocgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (rocgraph_byte_t*)h_vertices.data(), vertices, &ret_error),
        ret_error);

    CHECK_ROCGRAPH_SUCCESS_ERROR(
        rocgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (rocgraph_byte_t*)h_distances.data(), distances, &ret_error),
        ret_error);

    CHECK_ROCGRAPH_SUCCESS_ERROR(
        rocgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (rocgraph_byte_t*)h_predecessors.data(), predecessors, &ret_error),
        ret_error);

    ROCGRAPH_CLIENTS_EXPECT_INDIRECT_ARRAY_EQ(num_vertices,
                                              expected_distances,
                                              1,
                                              (const vertex_t*)h_vertices.data(),
                                              h_distances.data(),
                                              1,
                                              (const vertex_t*)nullptr);

    ROCGRAPH_CLIENTS_EXPECT_INDIRECT_ARRAY_EQ(num_vertices,
                                              expected_predecessors,
                                              1,
                                              (const vertex_t*)h_vertices.data(),
                                              h_predecessors.data(),
                                              1,
                                              (const vertex_t*)nullptr);

    rocgraph_type_erased_device_array_free(p_sources);
    rocgraph_paths_result_free(p_result);
    rocgraph_sg_graph_free(p_graph);
    rocgraph_destroy_handle(p_handle);
    rocgraph_error_free(ret_error);
};

static void BfsExceptions(const Arguments& arg)
{
    rocgraph_error_t* ret_error = nullptr;

    size_t num_edges = 8;
    // size_t num_vertices = 6;
    size_t depth_limit = 1;
    size_t num_seeds   = 1;

    int32_t src[]   = {0, 1, 1, 2, 2, 2, 3, 4};
    int32_t dst[]   = {1, 3, 4, 0, 1, 3, 5, 5};
    float_t wgt[]   = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
    int64_t seeds[] = {0};

    rocgraph_handle_t*                        p_handle      = nullptr;
    rocgraph_graph_t*                         p_graph       = nullptr;
    rocgraph_paths_result_t*                  p_result      = nullptr;
    rocgraph_type_erased_device_array_t*      p_sources     = nullptr;
    rocgraph_type_erased_device_array_view_t* p_source_view = nullptr;

    CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));
    CHECK_ROCGRAPH_SUCCESS((p_handle != nullptr) ? rocgraph_status_success
                                                 : rocgraph_status_invalid_handle);

    rocgraph_clients_create_test_graph(p_handle,
                                       src,
                                       dst,
                                       wgt,
                                       num_edges,
                                       rocgraph_bool_false,
                                       rocgraph_bool_false,
                                       rocgraph_bool_false,
                                       &p_graph,
                                       &ret_error);

    /*
   * FIXME: in create_graph_test.c, variables are defined but then hard-coded to
   * the constant rocgraph_data_type_id_int32. It would be better to pass the types into the functions
   * in both cases so that the test cases could be parameterized in the main.
   */
    CHECK_ROCGRAPH_SUCCESS_ERROR(
        rocgraph_type_erased_device_array_create(
            p_handle, num_seeds, rocgraph_data_type_id_int64, &p_sources, &ret_error),
        ret_error);

    p_source_view = rocgraph_type_erased_device_array_view(p_sources);

    CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                     p_handle, p_source_view, (rocgraph_byte_t*)seeds, &ret_error),
                                 ret_error);

    CHECK_ROCGRAPH_STATUS(rocgraph_bfs(p_handle,
                                       p_graph,
                                       p_source_view,
                                       rocgraph_bool_false,
                                       depth_limit,
                                       rocgraph_bool_true,
                                       rocgraph_bool_false,
                                       &p_result,
                                       &ret_error),
                          rocgraph_status_invalid_input);
}

static void Bfs(const Arguments& arg)
{
    static constexpr rocgraph_bool s_store_transposed = rocgraph_bool_false;
    static constexpr size_t        num_edges          = 8;
    static constexpr size_t        num_vertices       = 6;

    int32_t src[]                   = {0, 1, 1, 2, 2, 2, 3, 4};
    int32_t dst[]                   = {1, 3, 4, 0, 1, 3, 5, 5};
    float   wgt[]                   = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
    int32_t seeds[]                 = {0};
    int32_t expected_distances[]    = {0, 1, 2147483647, 2, 2, 3};
    int32_t expected_predecessors[] = {-1, 0, -1, 1, 1, 3};

    // Bfs wants store_transposed = rocgraph_bool_false
    generic_bfs_test(src,
                     dst,
                     wgt,
                     seeds,
                     expected_distances,
                     expected_predecessors,
                     num_vertices,
                     num_edges,
                     1,
                     10,
                     s_store_transposed);
}

static void BfsWithTranspose(const Arguments& arg)
{
    static constexpr rocgraph_bool s_store_transposed = rocgraph_bool_true;
    static constexpr size_t        num_edges          = 8;
    static constexpr size_t        num_vertices       = 6;

    int32_t src[]                   = {0, 1, 1, 2, 2, 2, 3, 4};
    int32_t dst[]                   = {1, 3, 4, 0, 1, 3, 5, 5};
    float   wgt[]                   = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
    int32_t seeds[]                 = {0};
    int32_t expected_distances[]    = {0, 1, 2147483647, 2, 2, 3};
    int32_t expected_predecessors[] = {-1, 0, -1, 1, 1, 3};

    // Bfs wants store_transposed = rocgraph_bool_false
    //    This call will force rocgraph_bfs to transpose the graph
    generic_bfs_test(src,
                     dst,
                     wgt,
                     seeds,
                     expected_distances,
                     expected_predecessors,
                     num_vertices,
                     num_edges,
                     1,
                     10,
                     s_store_transposed);
}

template <typename T>
void testing_rocgraph_bfs_bad_arg(const Arguments& arg)
{
    const rocgraph_handle_t*                  handle{};
    rocgraph_graph_t*                         graph{};
    rocgraph_type_erased_device_array_view_t* sources{};
    rocgraph_bool                             direction_optimizing{};
    size_t                                    depth_limit{};
    rocgraph_bool                             compute_predecessors{};
    rocgraph_bool                             do_expensive_check{};
    rocgraph_paths_result_t**                 result{};
    rocgraph_error_t**                        error{};
    auto                                      ret = rocgraph_bfs(handle,
                            graph,
                            sources,
                            direction_optimizing,
                            depth_limit,
                            compute_predecessors,
                            do_expensive_check,
                            result,
                            error);
    if(ret != rocgraph_status_success)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_internal_error);
    }
    CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
}

template <typename T>
void testing_rocgraph_bfs(const Arguments& arg)
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

#define INSTANTIATE(TYPE)                                                   \
    template void testing_rocgraph_bfs_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_rocgraph_bfs<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_bfs_extra(const Arguments& arg)
{
    testing_dispatch_extra(
        arg, "BfsWithTranspose", BfsWithTranspose, "Bfs", Bfs, "BfsExceptions", BfsExceptions);
}
