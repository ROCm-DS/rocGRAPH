/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "testing_rocgraph_overlap_coefficients.hpp"
#include "rocgraph/rocgraph.h"
#include "similarity_test.hpp"
#include "testing.hpp"
static void Overlap(const Arguments& arg)
{

    using vertex_t = int32_t;

    using weight_t      = float;
    size_t num_edges    = 16;
    size_t num_vertices = 6;
    size_t num_pairs    = 10;

    vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
    vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
    weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
    vertex_t h_first[]  = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3};
    vertex_t h_second[] = {1, 3, 4, 2, 3, 5, 3, 4, 5, 4};
    weight_t h_result[] = {0.5, 1, 0.5, 0.666667, 0.333333, 1, 0.333333, 0.5, 0.5, 1};

    generic_similarity_test<weight_t, vertex_t>(h_src,
                                                h_dst,
                                                h_wgt,
                                                h_first,
                                                h_second,
                                                h_result,
                                                num_vertices,
                                                num_edges,
                                                num_pairs,
                                                rocgraph_bool_false,
                                                rocgraph_bool_false,
                                                OVERLAP);
}

static void WeightedOverlap(const Arguments& arg)
{

    using vertex_t = int32_t;

    using weight_t      = float;
    size_t num_edges    = 16;
    size_t num_vertices = 7;
    size_t num_pairs    = 3;

    vertex_t h_src[] = {0, 1, 2, 0, 1, 2, 3, 3, 3, 4, 4, 4, 0, 5, 2, 6};
    vertex_t h_dst[] = {3, 3, 3, 4, 4, 4, 0, 1, 2, 0, 1, 2, 5, 0, 6, 2};
    weight_t h_wgt[]
        = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5, 4.0, 4.0};

    vertex_t h_first[]  = {0, 0, 1};
    vertex_t h_second[] = {1, 2, 3};
    weight_t h_result[] = {0.714286, 0.416667, 0.000000};

    generic_similarity_test<weight_t, vertex_t>(h_src,
                                                h_dst,
                                                h_wgt,
                                                h_first,
                                                h_second,
                                                h_result,
                                                num_vertices,
                                                num_edges,
                                                num_pairs,
                                                rocgraph_bool_false,
                                                rocgraph_bool_true,
                                                OVERLAP);
}

static void AllPairsOverlap(const Arguments& arg)
{

    using vertex_t = int32_t;

    using weight_t      = float;
    size_t num_edges    = 16;
    size_t num_vertices = 6;
    size_t num_pairs    = 22;

    vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
    vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
    weight_t h_wgt[] = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};

    vertex_t h_first[]  = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5};
    vertex_t h_second[] = {1, 2, 3, 4, 0, 2, 3, 5, 0, 1, 3, 4, 5, 0, 1, 2, 4, 0, 2, 3, 1, 2};
    weight_t h_result[]
        = {0.5, 0.5, 1, 0.5,      0.5,      0.666667, 0.333333, 1,   0.5, 0.666667, 0.333333,
           0.5, 0.5, 1, 0.333333, 0.333333, 1,        0.5,      0.5, 1,   1,        0.5};

    generic_all_pairs_similarity_test<weight_t, vertex_t>(h_src,
                                                          h_dst,
                                                          h_wgt,
                                                          h_first,
                                                          h_second,
                                                          h_result,
                                                          num_vertices,
                                                          num_edges,
                                                          num_pairs,
                                                          rocgraph_bool_false,
                                                          rocgraph_bool_false,
                                                          SIZE_MAX,
                                                          OVERLAP);
}

static void WeightedAllPairsOverlap(const Arguments& arg)
{

    using vertex_t = int32_t;

    using weight_t      = float;
    size_t num_edges    = 16;
    size_t num_vertices = 7;
    size_t num_pairs    = 16;

    vertex_t h_src[] = {0, 1, 2, 0, 1, 2, 3, 3, 3, 4, 4, 4, 0, 5, 2, 6};
    vertex_t h_dst[] = {3, 3, 3, 4, 4, 4, 0, 1, 2, 0, 1, 2, 5, 0, 6, 2};
    weight_t h_wgt[]
        = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5, 4.0, 4.0};

    vertex_t h_first[]  = {0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6};
    vertex_t h_second[] = {1, 2, 0, 2, 0, 1, 4, 5, 6, 3, 5, 6, 3, 4, 3, 4};
    weight_t h_result[] = {0.714286,
                           0.416667,
                           0.714286,
                           1,
                           0.416667,
                           1,
                           1,
                           0.166667,
                           0.5,
                           1,
                           0.571429,
                           0.75,
                           0.166667,
                           0.571429,
                           0.5,
                           0.75};

    generic_all_pairs_similarity_test<weight_t, vertex_t>(h_src,
                                                          h_dst,
                                                          h_wgt,
                                                          h_first,
                                                          h_second,
                                                          h_result,
                                                          num_vertices,
                                                          num_edges,
                                                          num_pairs,
                                                          rocgraph_bool_false,
                                                          rocgraph_bool_true,
                                                          SIZE_MAX,
                                                          OVERLAP);
}

static void AllPairsOverlapTopk(const Arguments& arg)
{

    using vertex_t = int32_t;

    using weight_t      = float;
    size_t num_edges    = 16;
    size_t num_vertices = 6;
    size_t num_pairs    = 6;
    size_t topk         = 6;

    vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
    vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
    weight_t h_wgt[] = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};

    vertex_t h_first[]  = {0, 1, 3, 3, 4, 5};
    vertex_t h_second[] = {3, 5, 0, 4, 3, 1};
    weight_t h_result[] = {1, 1, 1, 1, 1, 1};

    generic_all_pairs_similarity_test<weight_t, vertex_t>(h_src,
                                                          h_dst,
                                                          h_wgt,
                                                          h_first,
                                                          h_second,
                                                          h_result,
                                                          num_vertices,
                                                          num_edges,
                                                          num_pairs,
                                                          rocgraph_bool_false,
                                                          rocgraph_bool_false,
                                                          topk,
                                                          OVERLAP);
}

static void WeightedAllPairsOverlapTopk(const Arguments& arg)
{

    using vertex_t      = int32_t;
    using weight_t      = float;
    size_t num_edges    = 16;
    size_t num_vertices = 7;
    size_t num_pairs    = 6;
    size_t topk         = 6;

    vertex_t h_src[] = {0, 1, 2, 0, 1, 2, 3, 3, 3, 4, 4, 4, 0, 5, 2, 6};
    vertex_t h_dst[] = {3, 3, 3, 4, 4, 4, 0, 1, 2, 0, 1, 2, 5, 0, 6, 2};
    weight_t h_wgt[]
        = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5, 4.0, 4.0};

    vertex_t h_first[]  = {1, 2, 3, 4, 4, 6};
    vertex_t h_second[] = {2, 1, 4, 3, 6, 4};
    weight_t h_result[] = {1, 1, 1, 1, 0.75, 0.75};

    generic_all_pairs_similarity_test<weight_t, vertex_t>(h_src,
                                                          h_dst,
                                                          h_wgt,
                                                          h_first,
                                                          h_second,
                                                          h_result,
                                                          num_vertices,
                                                          num_edges,
                                                          num_pairs,
                                                          rocgraph_bool_false,
                                                          rocgraph_bool_true,
                                                          topk,
                                                          OVERLAP);
}

template <typename T>
void testing_rocgraph_overlap_coefficients_bad_arg(const Arguments& arg)
{
#ifdef TO_FILL

    const rocgraph_handle_t*       handle{};
    rocgraph_graph_t*              graph{};
    const rocgraph_vertex_pairs_t* vertex_pairs{};
    rocgraph_bool                  use_weight{};
    rocgraph_bool                  do_expensive_check{};
    rocgraph_similarity_result_t** result{};
    rocgraph_error_t**             error{};
    auto                           ret = rocgraph_overlap_coefficients(
        handle, graph, vertex_pairs, use_weight, do_expensive_check, result, error);
    if(ret != rocgraph_status_success)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_internal_error);
    }
    CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
#endif
}

template <typename T>
void testing_rocgraph_overlap_coefficients(const Arguments& arg)
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
    template void testing_rocgraph_overlap_coefficients_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_rocgraph_overlap_coefficients<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_overlap_coefficients_extra(const Arguments& arg)
{
    testing_dispatch_extra(arg,
                           "Overlap",
                           Overlap,
                           "WeightedOverlap",
                           WeightedOverlap,
                           "AllPairsOverlap",
                           AllPairsOverlap,
                           "WeightedAllPairsOverlap",
                           WeightedAllPairsOverlap,
                           "AllPairsOverlapTopk",
                           AllPairsOverlapTopk,
                           "WeightedAllPairsOverlapTopk",
                           WeightedAllPairsOverlapTopk);
}
