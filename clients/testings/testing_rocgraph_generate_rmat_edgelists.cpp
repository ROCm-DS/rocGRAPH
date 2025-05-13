/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "testing_rocgraph_generate_rmat_edgelists.hpp"
#include "rocgraph/rocgraph.h"
#include "testing.hpp"

template <typename T>
void testing_rocgraph_generate_rmat_edgelists_bad_arg(const Arguments& arg)
{
    const rocgraph_handle_t*        handle{};
    rocgraph_rng_state_t*           rng_state{};
    size_t                          n_edgelists{};
    size_t                          min_scale{};
    size_t                          max_scale{};
    size_t                          edge_factor{};
    rocgraph_generator_distribution size_distribution{};
    rocgraph_generator_distribution edge_distribution{};
    rocgraph_bool                   clip_and_flip{};
    rocgraph_bool                   scramble_vertex_ids{};
    rocgraph_coo_list_t**           result{};
    rocgraph_error_t**              error{};
    auto                            ret = rocgraph_generate_rmat_edgelists(handle,
                                                rng_state,
                                                n_edgelists,
                                                min_scale,
                                                max_scale,
                                                edge_factor,
                                                size_distribution,
                                                edge_distribution,
                                                clip_and_flip,
                                                scramble_vertex_ids,
                                                result,
                                                error);
    if(ret != rocgraph_status_success)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_internal_error);
    }
    CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
}

template <typename T>
void testing_rocgraph_generate_rmat_edgelists(const Arguments& arg)
{
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
}

#define INSTANTIATE(TYPE)                                                                       \
    template void testing_rocgraph_generate_rmat_edgelists_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_rocgraph_generate_rmat_edgelists<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_generate_rmat_edgelists_extra(const Arguments& arg) {}
