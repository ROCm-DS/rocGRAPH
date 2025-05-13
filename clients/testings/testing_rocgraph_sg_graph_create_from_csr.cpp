/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "testing_rocgraph_sg_graph_create_from_csr.hpp"
#include "rocgraph/rocgraph.h"
#include "testing.hpp"

template <typename T>
void testing_rocgraph_sg_graph_create_from_csr_bad_arg(const Arguments& arg)
{
    const rocgraph_handle_t*                        handle{};
    const rocgraph_graph_properties_t*              properties{};
    const rocgraph_type_erased_device_array_view_t* offsets{};
    const rocgraph_type_erased_device_array_view_t* indices{};
    const rocgraph_type_erased_device_array_view_t* weights{};
    const rocgraph_type_erased_device_array_view_t* edge_ids{};
    const rocgraph_type_erased_device_array_view_t* edge_type_ids{};
    rocgraph_bool                                   store_transposed{};
    rocgraph_bool                                   renumber{};
    rocgraph_bool                                   do_expensive_check{};
    rocgraph_graph_t**                              graph{};
    rocgraph_error_t**                              error{};
    auto                                            ret = rocgraph_sg_graph_create_from_csr(handle,
                                                 properties,
                                                 offsets,
                                                 indices,
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
}

template <typename T>
void testing_rocgraph_sg_graph_create_from_csr(const Arguments& arg)
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

#define INSTANTIATE(TYPE)                                                                        \
    template void testing_rocgraph_sg_graph_create_from_csr_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_rocgraph_sg_graph_create_from_csr<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_sg_graph_create_from_csr_extra(const Arguments& arg) {}
