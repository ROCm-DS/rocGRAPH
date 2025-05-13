/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "testing_rocgraph_weakly_connected_components.hpp"
#include "rocgraph/rocgraph.h"
#include "testing.hpp"

#include "rocgraph_clients_create_test_graph.hpp"
#include "rocgraph_test.hpp"

namespace
{
    template <typename weight_t, typename vertex_t>
    static void generic_wcc_bench(vertex_t*     h_src,
                                  vertex_t*     h_dst,
                                  weight_t*     h_wgt,
                                  vertex_t*     h_result,
                                  size_t        num_vertices,
                                  size_t        num_edges,
                                  rocgraph_bool store_transposed)
    {
        rocgraph_error_t* ret_error;

        rocgraph_handle_t*          p_handle = nullptr;
        rocgraph_graph_t*           p_graph  = nullptr;
        rocgraph_labeling_result_t* p_result = nullptr;
        rocgraph_local_handle       local_handle;
        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        rocgraph_clients_create_test_graph(p_handle,
                                           h_src,
                                           h_dst,
                                           h_wgt,
                                           num_edges,
                                           store_transposed,
                                           rocgraph_bool_false,
                                           rocgraph_bool_true,
                                           &p_graph,
                                           &ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_weakly_connected_components(
                p_handle, p_graph, rocgraph_bool_false, &p_result, &ret_error),
            ret_error);

        rocgraph_labeling_result_free(p_result);
        rocgraph_sg_graph_free(p_graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    template <typename weight_t, typename vertex_t>
    static void generic_wcc_test(vertex_t*     h_src,
                                 vertex_t*     h_dst,
                                 weight_t*     h_wgt,
                                 vertex_t*     h_result,
                                 size_t        num_vertices,
                                 size_t        num_edges,
                                 rocgraph_bool store_transposed)
    {
        rocgraph_error_t* ret_error;

        rocgraph_handle_t*          p_handle = nullptr;
        rocgraph_graph_t*           p_graph  = nullptr;
        rocgraph_labeling_result_t* p_result = nullptr;

        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        rocgraph_clients_create_test_graph(p_handle,
                                           h_src,
                                           h_dst,
                                           h_wgt,
                                           num_edges,
                                           store_transposed,
                                           rocgraph_bool_false,
                                           rocgraph_bool_true,
                                           &p_graph,
                                           &ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_weakly_connected_components(
                p_handle, p_graph, rocgraph_bool_false, &p_result, &ret_error),
            ret_error);

        rocgraph_type_erased_device_array_view_t* vertices;
        rocgraph_type_erased_device_array_view_t* components;

        vertices   = rocgraph_labeling_result_get_vertices(p_result);
        components = rocgraph_labeling_result_get_labels(p_result);

        std::vector<vertex_t> h_vertices(num_vertices);
        std::vector<vertex_t> h_components(num_vertices);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_vertices.data(), vertices, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_components.data(), components, &ret_error),
            ret_error);

        std::vector<vertex_t> component_check(num_vertices, vertex_t(num_vertices));

        for(size_t i = 0; i < num_vertices; ++i)
        {
            if(component_check[h_result[h_vertices[i]]] == num_vertices)
            {
                component_check[h_result[h_vertices[i]]] = h_components[i];
            }
        }

        for(size_t i = 0; i < num_vertices; ++i)
        {
            ROCGRAPH_CLIENTS_EXPECT_EQ(h_components[i], component_check[h_result[h_vertices[i]]]);
        }

        rocgraph_type_erased_device_array_view_free(components);
        rocgraph_type_erased_device_array_view_free(vertices);
        rocgraph_labeling_result_free(p_result);
        rocgraph_sg_graph_free(p_graph);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    template <typename T>
    static void WeaklyConnectedComponents(const Arguments& arg)
    {
        using vertex_t      = int32_t;
        using weight_t      = T;
        size_t num_edges    = 32;
        size_t num_vertices = 12;

        vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 6, 7, 7,  8, 8, 8, 9,  10,
                            1, 3, 4, 0, 1, 3, 5, 5, 7, 9, 10, 6, 7, 9, 11, 11};
        vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 7, 9, 10, 6, 7, 9, 11, 11,
                            0, 1, 1, 2, 2, 2, 3, 4, 6, 7, 7,  8, 8, 8, 9,  10};
        weight_t h_wgt[]
            = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        vertex_t h_result[] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};

        // WCC wants store_transposed = rocgraph_bool_false
        generic_wcc_test(
            h_src, h_dst, h_wgt, h_result, num_vertices, num_edges, rocgraph_bool_false);
    }

    template <typename T>
    static void WeaklyConnectedComponentsTranspose(const Arguments& arg)
    {
        using vertex_t      = int32_t;
        using weight_t      = T;
        size_t num_edges    = 32;
        size_t num_vertices = 12;

        vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 6, 7, 7,  8, 8, 8, 9,  10,
                            1, 3, 4, 0, 1, 3, 5, 5, 7, 9, 10, 6, 7, 9, 11, 11};
        vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 7, 9, 10, 6, 7, 9, 11, 11,
                            0, 1, 1, 2, 2, 2, 3, 4, 6, 7, 7,  8, 8, 8, 9,  10};
        weight_t h_wgt[]
            = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        vertex_t h_result[] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};

        // WCC wants store_transposed = rocgraph_bool_false
        generic_wcc_test(
            h_src, h_dst, h_wgt, h_result, num_vertices, num_edges, rocgraph_bool_true);
    }

    template <typename T>
    static void WeaklyConnectedComponentsTranspose_main(const Arguments& arg)
    {
        int64_t M   = arg.M;
        int64_t NNZ = M; //random_cached_generator_exact<int64_t>(0,3*M);

        // std::cout << M << std::endl;
        // std::cout << NNZ << std::endl;

        using vertex_t      = int32_t;
        using weight_t      = T;
        size_t num_edges    = NNZ;
        size_t num_vertices = M;

        std::vector<vertex_t> h_src(num_edges);
        std::vector<vertex_t> h_dst(num_edges);
        std::vector<weight_t> h_wgt(num_edges, 1.0);
        std::vector<vertex_t> h_result(num_vertices, 0);
        size_t                k;
        for(k = 0; k < M - 4; ++k)
        {
            vertex_t i   = k;
            vertex_t j   = (k + 4) % M;
            h_src[k + 4] = std::min(i, j);
            h_dst[k + 4] = std::max(i, j);
        }
        for(k = M - 4; k < M; ++k)
        {
            vertex_t i   = k;
            vertex_t j   = (k + 4) % M;
            h_src[M - k] = std::min(i, j);
            h_dst[M - k] = std::max(i, j);
        }
        // WCC wants store_transposed = rocgraph_bool_false
        generic_wcc_bench(h_src.data(),
                          h_dst.data(),
                          h_wgt.data(),
                          h_result.data(),
                          num_vertices,
                          num_edges,
                          rocgraph_bool_true);
    }

    template <typename T>
    static void WeaklyConnectedComponents_main(const Arguments& arg)
    {
        int64_t M   = arg.M;
        int64_t NNZ = M; //random_cached_generator_exact<int64_t>(0,3*M);

        // std::cout << M << std::endl;
        // std::cout << NNZ << std::endl;

        using vertex_t      = int32_t;
        using weight_t      = T;
        size_t num_edges    = NNZ;
        size_t num_vertices = M;

        std::vector<vertex_t> h_src(num_edges);
        std::vector<vertex_t> h_dst(num_edges);
        std::vector<weight_t> h_wgt(num_edges, 1.0);
        std::vector<vertex_t> h_result(num_vertices, 0);
        size_t                k;
        for(k = 0; k < M - 4; ++k)
        {
            vertex_t i   = k;
            vertex_t j   = (k + 4) % M;
            h_src[k + 4] = std::min(i, j);
            h_dst[k + 4] = std::max(i, j);
        }
        for(k = M - 4; k < M; ++k)
        {
            vertex_t i   = k;
            vertex_t j   = (k + 4) % M;
            h_src[M - k] = std::min(i, j);
            h_dst[M - k] = std::max(i, j);
        }
#if 0
    for (;k<num_edges;++k)
      {
	bool f = true;
	while (f)
	  {
	    f=false;
	    vertex_t i = random_cached_generator_exact<vertex_t>(0,num_vertices-1);
	    vertex_t j = random_cached_generator_exact<vertex_t>(0,num_vertices-1);
	    while (i==j)
	      {
		j = random_cached_generator_exact<vertex_t>(0,num_vertices-1);
	      }
	    for (size_t p=0;p<k;++p)
	      {
		if (h_src[p] == i &&    h_dst[p] == j)
		  {
		    f=true;
		    break;
		  }
	      }
	    //	    std::cout <<"(" << i << " , " << j  << ")"<< std::endl;
	  }
      }
    for (k=0;k<num_edges;++k)
      {
	std::cout <<"(" << h_src[k] << " , " << h_dst[k]  << ")"<< std::endl;
      }
#endif
        // WCC wants store_transposed = rocgraph_bool_false
        generic_wcc_bench(h_src.data(),
                          h_dst.data(),
                          h_wgt.data(),
                          h_result.data(),
                          num_vertices,
                          num_edges,
                          rocgraph_bool_false);
    }

} // namespace

template <typename T>
void testing_rocgraph_weakly_connected_components_bad_arg(const Arguments& arg)
{
#ifdef TO_FILL

    const rocgraph_handle_t*     handle{};
    rocgraph_graph_t*            graph{};
    rocgraph_bool                do_expensive_check{};
    rocgraph_labeling_result_t** result{};
    rocgraph_error_t**           error{};
    auto                         ret
        = rocgraph_weakly_connected_components(handle, graph, do_expensive_check, result, error);
    if(ret != rocgraph_status_success)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_internal_error);
    }
    CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
#endif
}

template <typename T>
void testing_rocgraph_weakly_connected_components(const Arguments& arg)
{
    WeaklyConnectedComponents_main<T>(arg);

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

#define INSTANTIATE(TYPE)                                                     \
    template void testing_rocgraph_weakly_connected_components_bad_arg<TYPE>( \
        const Arguments& arg);                                                \
    template void testing_rocgraph_weakly_connected_components<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_weakly_connected_components_extra(const Arguments& arg)
{
    if(arg.a_type == rocgraph_datatype_f32_r)
    {
        testing_dispatch_extra(arg,
                               "WeaklyConnectedComponents",
                               WeaklyConnectedComponents<float>,
                               "WeaklyConnectedComponentsTranspose",
                               WeaklyConnectedComponentsTranspose<float>);
    }
    else if(arg.a_type == rocgraph_datatype_f64_r)
    {
        testing_dispatch_extra(arg,
                               "WeaklyConnectedComponents",
                               WeaklyConnectedComponents<double>,
                               "WeaklyConnectedComponentsTranspose",
                               WeaklyConnectedComponentsTranspose<double>);
    }
}
